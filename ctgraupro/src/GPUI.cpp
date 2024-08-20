#include "unistd.h"
#include <algorithm>  // for std::sort

// PCL utilities
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

// ROS utilities
#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "livox_ros_driver/CustomMsg.h"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "utility.h"
#include "GaussianProcess.hpp"
#include "GPMUI.hpp"

// Topics
#include "cf_msgs/Tdoa.h"
#include "cf_msgs/Tof.h"

using namespace std;

boost::shared_ptr<ros::NodeHandle> nh_ptr;

template <typename Scalar = double, int RowSize = Dynamic, int ColSize = Dynamic>
Matrix<Scalar, RowSize, ColSize> load_dlm(const std::string &path, string dlm, int r_start = 0, int col_start = 0)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int row_idx = -1;
    int rows = 0;
    while (std::getline(indata, line))
    {
        row_idx++;
        if (row_idx < r_start)
            continue;

        std::stringstream lineStream(line);
        std::string cell;
        int col_idx = -1;
        while (std::getline(lineStream, cell, dlm[0]))
        {
            if (cell == dlm || cell.size() == 0)
                continue;

            col_idx++;
            if (col_idx < col_start)
                continue;

            values.push_back(std::stod(cell));
        }

        rows++;
    }

    return Map<const Matrix<Scalar, RowSize, ColSize, RowMajor>>(values.data(), rows, values.size() / rows);
}

std::map<uint16_t, Eigen::Vector3d> getAnchorListFromUTIL(const std::string& anchor_path)
{
    std::map<uint16_t, Eigen::Vector3d> anchor_list;
    std::string line;
    std::ifstream infile;
    infile.open(anchor_path);
    if (!infile) {
        std::cerr << "Unable to open file: " << anchor_path << std::endl;
        exit(1);
    }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        char comma, tmp, tmp2;
        int anchor_id;
        double x, y, z;
        iss >> tmp >> tmp >> anchor_id >> tmp >> tmp2 >> comma >> x >> comma >> y >> comma >> z;
        if (tmp2 == 'p') {
            anchor_list[anchor_id] = Eigen::Vector3d(x, y, z);
        }
    }
    infile.close();
    return anchor_list;
}

typedef sensor_msgs::Imu  ImuMsg    ;
typedef ImuMsg::ConstPtr  ImuMsgPtr ;
typedef cf_msgs::Tdoa     TdoaMsg   ;
typedef TdoaMsg::ConstPtr TdoaMsgPtr;
typedef cf_msgs::Tof      TofMsg    ;
typedef TofMsg::ConstPtr  TofMsgPtr ;

const double POSINF =  std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

vector<SE3d> anc_pose;

double gpDt = 0.02;
Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
int  WINDOW_SIZE = 4;
int  SLIDE_SIZE  = 2;
double w_tdoa = 0.1;

GaussianProcessPtr traj;

bool fuse_tdoa = true;
bool fuse_tof  = false;
bool fuse_imu  = true;

struct UwbImuBuf
{
    deque<TdoaMsgPtr> tdoaBuf;
    deque<TofMsgPtr>  tofBuf;
    deque<ImuMsgPtr>  imuBuf;

    mutex tdoaBuf_mtx;
    mutex tofBuf_mtx;
    mutex imuBuf_mtx;

    vector<TDOAData> tdoa_data;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (tdoaBuf.size() != 0 && fuse_tdoa)
            tmin = min(tmin, tdoaBuf.front()->header.stamp.toSec());
        if (tofBuf.size() != 0 && fuse_tof)
            tmin = min(tmin, tofBuf.front()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmin = min(tmin, imuBuf.front()->header.stamp.toSec());
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (tdoaBuf.size() != 0 && fuse_tdoa)
            tmax = max(tmax, tdoaBuf.back()->header.stamp.toSec());
        if (tofBuf.size() != 0 && fuse_tof)
            tmax = max(tmax, tofBuf.back()->header.stamp.toSec());
        if (imuBuf.size() != 0 && fuse_imu)
            tmax = max(tmax, imuBuf.back()->header.stamp.toSec());
        return tmax;
    }

    template<typename T>
    void transferDataOneBuf(deque<T> &selfbuf, deque<T> &otherbuf, mutex &otherbufmtx, double tmax)
    {
        while(otherbuf.size() != 0)
        {
            if (otherbuf.front()->header.stamp.toSec() <= tmax)
            {
                lock_guard<mutex> lg(otherbufmtx);
                selfbuf.push_back(otherbuf.front());
                otherbuf.pop_front();
            }
            else
                break;
        }
    }

    void transferData(UwbImuBuf &other, double tmax)
    {
        if (fuse_tdoa) transferDataOneBuf(tdoaBuf, other.tdoaBuf, other.tdoaBuf_mtx, tmax);
        if (fuse_tof ) transferDataOneBuf(tofBuf,  other.tofBuf,  other.tofBuf_mtx,  tmax);
        if (fuse_imu ) transferDataOneBuf(imuBuf,  other.imuBuf,  other.imuBuf_mtx,  tmax);
        transferTDOAData();
    }

    void transferTDOAData()
    {
        tdoa_data.clear();
        for (const auto& data : tdoaBuf) {
            TDOAData tdoa(data->header.stamp.toSec(), data->idA, data->idB, data->data);
            tdoa_data.push_back(tdoa);
        }
    }

    template<typename T>
    void slideForwardOneBuf(deque<T> &buf, double tremove)
    {
        while(buf.size() != 0)
            if(buf.front()->header.stamp.toSec() < tremove)
                buf.pop_front();
            else
                break;
    }

    void slideForward(double tremove)
    {
        if (fuse_tdoa) slideForwardOneBuf(tdoaBuf, tremove);
        if (fuse_tof ) slideForwardOneBuf(tofBuf,  tremove);
        if (fuse_imu ) slideForwardOneBuf(imuBuf,  tremove);
    }
};
UwbImuBuf UIBuf;
vector<Eigen::Vector3d> gtBuf;
nav_msgs::Path est_path;

ros::Publisher gt_pub;
ros::Publisher est_pub;

ros::Subscriber tdoaSub;
ros::Subscriber tofSub ;
ros::Subscriber imuSub ;
ros::Subscriber gtSub  ;

void tdoaCb(const TdoaMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.tdoaBuf_mtx);
    UIBuf.tdoaBuf.push_back(msg);
    // printf(KCYN "Receive tdoa\n" RESET);
}

void tofCb(const TofMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.tofBuf_mtx);
    UIBuf.tofBuf.push_back(msg);
    // printf(KBLU "Receive tof\n" RESET);
}

void imuCb(const ImuMsgPtr &msg)
{
    lock_guard<mutex> lg(UIBuf.imuBuf_mtx);
    UIBuf.imuBuf.push_back(msg);
    // printf(KMAG "Receive imu\n" RESET);
}

void gtCb(const geometry_msgs::PoseWithCovarianceStampedConstPtr& gt_msg)
{
    Eigen::Quaterniond q(gt_msg->pose.pose.orientation.w, gt_msg->pose.pose.orientation.x,
                            gt_msg->pose.pose.orientation.y, gt_msg->pose.pose.orientation.z);
    Eigen::Vector3d pos(gt_msg->pose.pose.position.x, gt_msg->pose.pose.position.y, gt_msg->pose.pose.position.z);
    gtBuf.push_back(pos);    

    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = "map";

    odom_msg.pose.pose.position.x = pos[0];
    odom_msg.pose.pose.position.y = pos[1];
    odom_msg.pose.pose.position.z = pos[2];

    odom_msg.pose.pose.orientation.w = q.w();
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    gt_pub.publish(odom_msg);        
}

ros::Publisher knot_pub;

void processData(GaussianProcessPtr traj, GPMUIPtr gpmui, std::map<uint16_t, Eigen::Vector3d> anchor_list)
{
    UwbImuBuf swUIBuf;

    // Loop and optimize
    while(ros::ok())
    {
        // Step 0: Check if there is data that can be admitted to the sw buffer
        double newMaxTime = traj->getMaxTime() + SLIDE_SIZE*gpDt;

        ros::Time timeout = ros::Time::now();
        if(UIBuf.maxTime() < newMaxTime)
        {
            if(auto_exit && (ros::Time::now() - timeout).toSec() > 20.0)
            {
                printf("Polling time out exiting.\n");
                exit(-1);
            }
            static int msWait = int(SLIDE_SIZE*gpDt*1000);
            this_thread::sleep_for(chrono::milliseconds(msWait));
            continue;
        }
        timeout = ros::Time::now();

        // Step 1: Extract the data to the local buffer
        swUIBuf.transferData(UIBuf, newMaxTime);

        // Step 2: Extend the trajectory
        if (traj->getMaxTime() < newMaxTime && (newMaxTime - traj->getMaxTime()) > gpDt*0.01) {
            traj->extendOneKnot();
        }

        // Step 3: Optimization
        TicToc tt_solve;
        
        int outer_iter = 1;
        
        // double tmin = swUIBuf.tdoa_data.front().t;     // Start time of the sliding window
        // double tmax = swUIBuf.tdoa_data.back().t;      // End time of the sliding window      
        double tmin = traj->getKnotTime(traj->getNumKnots() - WINDOW_SIZE) + 1e-3;     // Start time of the sliding window
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3;      // End time of the sliding window              
        double tmid = tmin + SLIDE_SIZE*traj->getDt() + 1e-3;     // Next start time of the sliding window,
                                               // also determines the marginalization time limit          
        gpmui->Evaluate(outer_iter, traj, tmin, tmax, tmid, swUIBuf.tdoa_data, anchor_list, traj->getNumKnots() >= WINDOW_SIZE, w_tdoa);
        tt_solve.Toc();
        std::cout << "swUIBuf.tdoa_data: " << swUIBuf.tdoa_data.size() << " tmin: " << tmin << " tmax: " << tmax << std::endl;
        // Step 4: Report, visualize
        printf("Traj: %f. Sw: %.3f -> %.3f. Buf: %d, %d, %d. Num knots: %d\n",
                traj->getMaxTime(), swUIBuf.minTime(), swUIBuf.maxTime(),
                UIBuf.tdoaBuf.size(), UIBuf.tofBuf.size(), UIBuf.imuBuf.size(), traj->getNumKnots());

        pcl::PointCloud<pcl::PointXYZ> est_knots;
        for (int i = 0; i < traj->getNumKnots(); i++) {   
            Eigen::Vector3d knot_pos = traj->getKnotPose(i).translation();
            est_knots.points.push_back(pcl::PointXYZ(knot_pos.x(), knot_pos.y(), knot_pos.z()));
        }
        sensor_msgs::PointCloud2 knot_msg;
        pcl::toROSMsg(est_knots, knot_msg);
        knot_msg.header.stamp = ros::Time::now();
        knot_msg.header.frame_id = "map";        
        knot_pub.publish(knot_msg);

        Eigen::Vector3d est_pos = traj->pose(swUIBuf.tdoa_data.front().t).translation();
        geometry_msgs::PoseStamped traj_msg;
        traj_msg.header.stamp = ros::Time::now();
        traj_msg.pose.position.x = est_pos.x();
        traj_msg.pose.position.y = est_pos.y();
        traj_msg.pose.position.z = est_pos.z();
        traj_msg.pose.orientation.w = 1;
        traj_msg.pose.orientation.x = 0;
        traj_msg.pose.orientation.y = 0;
        traj_msg.pose.orientation.z = 0;
        est_path.poses.push_back(traj_msg);
        est_pub.publish(est_path);

        // Step 5: Slide the window forward
        if (traj->getNumKnots() >= WINDOW_SIZE)
        {
            double removeTime = traj->getKnotTime(traj->getNumKnots() - WINDOW_SIZE + SLIDE_SIZE);
            std::cout << "removeTime: " << removeTime << std::endl;
            swUIBuf.slideForward(removeTime);
        }
    }
}

int main(int argc, char **argv)
{
    // Initialize the node
    ros::init(argc, argv, "gpui");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    // Determine if we exit if no data is received after a while
    bool auto_exit = Util::GetBoolParam(nh_ptr, "auto_exit", false);

    // Parameters for the GP trajectory
    double gpQr_ = 1.0, gpQc_ = 1.0;
    nh_ptr->getParam("gpDt", gpDt );
    nh_ptr->getParam("gpQr", gpQr_);
    nh_ptr->getParam("gpQc", gpQc_);
    gpQr = gpQr_*Matrix3d::Identity(3, 3);
    gpQc = gpQc_*Matrix3d::Identity(3, 3);

    // Find the path to anchor position
    string anchor_pose_path;
    nh_ptr->getParam("anchor_pose_path", anchor_pose_path);
    
    // Load the anchor pose 
    std::map<uint16_t, Eigen::Vector3d> anc_pose_ = getAnchorListFromUTIL(anchor_pose_path);
    // MatrixXd anc_pose_ = load_dlm(anchor_pose_path, ",", 1, 0);
    // for(int ridx = 0; ridx < anc_pose_.rows(); ridx++)
    // {
    //     Vector4d Q_; Q_ << anc_pose_.block<1, 4>(ridx, 3).transpose();
    //     Quaternd Q(Q_.w(), Q_.x(), Q_.y(), Q_.z());
    //     Vector3d P = anc_pose_.block<1, 3>(ridx, 0).transpose();
    //     anc_pose.push_back(SE3d(Q, P));

    //     myTf tf(anc_pose.back());
    //     printf("Anchor: %2d. Pos: %6.3f, %6.3f, %6.3f. Rot: %4.0f, %4.0f, %4.0f.\n",
    //             ridx, P.x(), P.y(), P.z(), tf.yaw(), tf.pitch(), tf.roll());
    // }

    // Topics to subscribe to
    string tdoa_topic; nh_ptr->getParam("tdoa_topic", tdoa_topic);
    string tof_topic;  nh_ptr->getParam("tof_topic", tof_topic);
    string imu_topic;  nh_ptr->getParam("imu_topic", imu_topic);
    string gt_topic;   nh_ptr->getParam("gt_topic", gt_topic);
    fuse_tdoa = Util::GetBoolParam(nh_ptr, "fuse_tdoa", fuse_tdoa);
    fuse_tof  = Util::GetBoolParam(nh_ptr, "fuse_tof" , fuse_tof );
    fuse_imu  = Util::GetBoolParam(nh_ptr, "fuse_imu" , fuse_imu );

    // Subscribe to the topics
    tdoaSub = nh_ptr->subscribe(tdoa_topic, 10, tdoaCb);
    tofSub  = nh_ptr->subscribe(tof_topic,  10, tofCb);
    imuSub  = nh_ptr->subscribe(imu_topic,  10, imuCb);
    gtSub   = nh_ptr->subscribe(gt_topic,   10, gtCb);

    // Publish estimates
    knot_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/estimated_knot", 10);
    gt_pub   = nh_ptr->advertise<nav_msgs::Odometry>("/ground_truth", 10);
    est_pub  = nh_ptr->advertise<nav_msgs::Path>("/estimated_trajectory", 10);

    est_path.header.frame_id = "map";

    // while(ros::ok())
    //     this_thread::sleep_for(chrono::milliseconds(100));

    // Time to check the buffers and perform optimization
    nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);
    nh_ptr->getParam("SLIDE_SIZE", SLIDE_SIZE);
    nh_ptr->getParam("w_tdoa", w_tdoa);

    // Create the trajectory
    traj = GaussianProcessPtr(new GaussianProcess(gpDt, gpQr, gpQc, true));
    GPMUIPtr gpmui(new GPMUI(nh_ptr));

    // Wait to get the initial time
    while(ros::ok())
    {
        if(UIBuf.minTime() == POSINF)
        {
            this_thread::sleep_for(chrono::milliseconds(5));
            ros::spinOnce();
            continue;
        }

        double t0 = UIBuf.minTime();
        traj->setStartTime(t0);

        // Set initial pose
        SE3d initial_pose;
        initial_pose.translation() = Eigen::Vector3d(1.25, 0.0, 0.07);
        traj->setKnot(0, GPState(t0, initial_pose));
        break;
    }
    printf(KGRN "Start time: %f\n" RESET, traj->getMinTime());

    // Start polling and processing the data
    thread pdthread(processData, traj, gpmui, anc_pose_);

    // Spin
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    pdthread.join();
    return 0;
}