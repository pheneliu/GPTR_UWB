#include "unistd.h"
#include <algorithm>  // for std::sort

// PCL utilities
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

#include <opencv2/opencv.hpp>

// ROS utilities
#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "livox_ros_driver/CustomMsg.h"


// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "utility.h"
#include "GaussianProcess.hpp"
#include "GPMVICalib.hpp"

using namespace std;

boost::shared_ptr<ros::NodeHandle> nh_ptr;

std::map<int, Eigen::Vector3d> getCornerPosition3D(const std::string& data_path)
{
    std::map<int, Eigen::Vector3d> corner_list;
    std::ifstream infile(data_path);
    std::string line;
    while (std::getline(infile, line))
    {
        int idx;
        double x,y,z;
        char comma;
        std::istringstream iss(line);
        iss >> idx >> comma >> x >> comma >> y >> comma >> z;
        Eigen::Vector3d pos(x, y, z);
        corner_list[idx] = pos;
    }
    infile.close();
    std::cout << "loaded " << corner_list.size() << " 3D positions of corners" << std::endl;
    return corner_list;
}

void getCornerPosition2D(const std::string& data_path, vector<CornerData> &corner_meas)
{
    corner_meas.clear();
    std::string line;
    std::ifstream infile;
    infile.open(data_path);
    if (!infile) {
        std::cerr << "Unable to open file: " << data_path << std::endl;
        exit(1);
    }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        double t_s, px, py;
        char comma;
        iss >> t_s >> comma;

        vector<Eigen::Vector2d> corners;
        vector<int> ids;
        int idx = 0;
        for (; iss >> px >> comma >> py; iss >> comma) {
            if (px < 0 || py < 0) {
                idx++;
                continue;
            }
            corners.push_back(Eigen::Vector2d(px, py));
            ids.push_back(idx);
            idx++;
        }
        corner_meas.emplace_back(t_s, ids, corners);
    }
    infile.close();
    std::cout << "loaded " << corner_meas.size() << " images wih corner positions" << std::endl;
}

void getCameraModel(const std::string& data_path, CameraCalibration &cam_calib)
{
    cv::FileStorage fsSettings(data_path, cv::FileStorage::READ);

    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings " << data_path << std::endl;
        return;
    }

    cv::FileNode root = fsSettings["value0"];
    cv::FileNode T_imu_cam = root["T_imu_cam"];
    cv::FileNode intrinsics = root["intrinsics"];
    cv::FileNode resolution = root["resolution"];


    for (int i = 0; i < 2; i++) {
        double x,y,z;
        x = T_imu_cam[i]["px"];
        y = T_imu_cam[i]["py"];
        z = T_imu_cam[i]["pz"];

        double qx, qy, qz, qw;
        qx = T_imu_cam[i]["qx"];
        qy = T_imu_cam[i]["qy"];
        qz = T_imu_cam[i]["qz"];
        qw = T_imu_cam[i]["qw"];

        double fx, fy, cx, cy, xi, alpha;
        fx = intrinsics[i]["fx"];
        fy = intrinsics[i]["fy"];
        cx = intrinsics[i]["cx"];
        cy = intrinsics[i]["cy"];
        xi = intrinsics[i]["xi"];
        alpha = intrinsics[i]["alpha"];

        Eigen::Quaterniond qic(qw, qx, qy, qz);
        Sophus::SE3d Tic;
        Tic.translation() = Eigen::Vector3d(x, y, z);
        Tic.so3() = Sophus::SO3d::fitToSO3(qic.toRotationMatrix());
        cam_calib.T_i_c.push_back(Tic);

        DoubleSphereCamera<double> intr;
        intr.setFromInit(fx, fy, cx, cy, xi, alpha);
        cam_calib.intrinsics.push_back(intr);

        // double w,v;
        // w = resolution[i][0];
        // w = resolution[i]["w"];
    }

    // std::cout << "22222222222 " << Eigen::Vector3d(x, y, z).transpose() << std::endl;
}

void getIMUMeasurements(const std::string &data_path, vector<IMUData> &imu_meas)
{
    imu_meas.clear();
    std::ifstream infile(data_path + "imu_data.csv");
    std::string line;
    while (std::getline(infile, line)) {
        if (line[0] == '#') continue;

        std::stringstream ss(line);

        char tmp;
        uint64_t timestamp;
        Eigen::Vector3d gyro, accel;

        ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >>
        tmp >> accel[0] >> tmp >> accel[1] >> tmp >> accel[2];

        double t_s = timestamp * 1e-9;

        IMUData imu(t_s, accel, gyro);
        imu_meas.push_back(imu);
    }
    infile.close();
    std::cout << "loaded " << imu_meas.size() << " IMU measurements" << std::endl;
}

void getGT(const std::string &data_path, nav_msgs::Path &gt_path)
{
    gt_path.poses.clear();

    std::ifstream infile(data_path + "cam_pose.csv");
    std::string line;
    while (std::getline(infile, line)) {
        if (line[0] == '#') continue;

        std::stringstream ss(line);

        char tmp;
        uint64_t timestamp;
        Eigen::Quaterniond q;
        Eigen::Vector3d pos;

        ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >>
            tmp >> q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z();

        geometry_msgs::PoseStamped traj_msg;
        traj_msg.header.stamp = ros::Time().fromNSec(timestamp);
        traj_msg.pose.position.x = pos.x();
        traj_msg.pose.position.y = pos.y();
        traj_msg.pose.position.z = pos.z();
        traj_msg.pose.orientation.w = q.w();
        traj_msg.pose.orientation.x = q.x();
        traj_msg.pose.orientation.y = q.y();
        traj_msg.pose.orientation.z = q.z();
        gt_path.poses.push_back(traj_msg);          

        // data->gt_timestamps.emplace_back(timestamp);
        // data->gt_pose_data.emplace_back(q, pos);
    }
    infile.close();
    std::cout << "loaded " << gt_path.poses.size() << " gt data" << std::endl;
}

// typedef sensor_msgs::Imu  ImuMsg    ;
// typedef ImuMsg::ConstPtr  ImuMsgPtr ;
// typedef cf_msgs::Tdoa     TdoaMsg   ;
// typedef TdoaMsg::ConstPtr TdoaMsgPtr;
// typedef cf_msgs::Tof      TofMsg    ;
// typedef TofMsg::ConstPtr  TofMsgPtr ;

const double POSINF =  std::numeric_limits<double>::infinity();
const double NEGINF = -std::numeric_limits<double>::infinity();

double gpDt = 0.02;
Matrix3d gpQr;
Matrix3d gpQc;

bool auto_exit;
int  WINDOW_SIZE = 4;
int  SLIDE_SIZE  = 2;
double w_corner = 0.1;
double GYR_N = 10;
double GYR_W = 10;
double ACC_N = 0.5;
double ACC_W = 10;
double corner_loss_thres = -1;
double mp_loss_thres = -1;

GaussianProcessPtr traj;

// bool fuse_tdoa = true;
// bool fuse_tof  = false;
// bool fuse_imu  = true;

bool acc_ratio = false;
bool gyro_unit = false;

struct CameraImuBuf
{
    // deque<TdoaMsgPtr> tdoaBuf;
    // deque<TofMsgPtr>  tofBuf;
    // deque<ImuMsgPtr>  imuBuf;

    // mutex tdoaBuf_mtx;
    // mutex tofBuf_mtx;
    // mutex imuBuf_mtx;

    vector<CornerData> corner_data_cam0;
    vector<CornerData> corner_data_cam1;
    vector<IMUData> imu_data;

    double minTime()
    {
        double tmin = std::numeric_limits<double>::infinity();
        if (corner_data_cam0.size() != 0)
            tmin = min(tmin, corner_data_cam0.front().t);
        if (corner_data_cam1.size() != 0)
            tmin = min(tmin, corner_data_cam1.front().t);
        if (imu_data.size() != 0)
            tmin = min(tmin, imu_data.front().t);
        return tmin;
    }

    double maxTime()
    {
        double tmax = -std::numeric_limits<double>::infinity();
        if (corner_data_cam0.size() != 0)
            tmax = max(tmax, corner_data_cam0.back().t);
        if (corner_data_cam1.size() != 0)
            tmax = max(tmax, corner_data_cam1.back().t);
        if (imu_data.size() != 0)
            tmax = max(tmax, imu_data.back().t);
        return tmax;
    }

    // template<typename T>
    // void transferDataOneBuf(deque<T> &selfbuf, deque<T> &otherbuf, mutex &otherbufmtx, double tmax)
    // {
    //     while(otherbuf.size() != 0)
    //     {
    //         if (otherbuf.front()->header.stamp.toSec() <= tmax)
    //         {
    //             lock_guard<mutex> lg(otherbufmtx);
    //             selfbuf.push_back(otherbuf.front());
    //             otherbuf.pop_front();
    //         }
    //         else
    //             break;
    //     }
    // }

    // void transferData(UwbImuBuf &other, double tmax)
    // {
    //     if (fuse_tdoa) transferDataOneBuf(tdoaBuf, other.tdoaBuf, other.tdoaBuf_mtx, tmax);
    //     if (fuse_tof ) transferDataOneBuf(tofBuf,  other.tofBuf,  other.tofBuf_mtx,  tmax);
    //     if (fuse_imu ) transferDataOneBuf(imuBuf,  other.imuBuf,  other.imuBuf_mtx,  tmax);
    //     transferTDOAData();
    //     if (fuse_imu ) {
    //         transferIMUData();
    //     }
    // }

    // void transferTDOAData()
    // {
    //     tdoa_data.clear();
    //     for (const auto& data : tdoaBuf) {
    //         TDOAData tdoa(data->header.stamp.toSec(), data->idA, data->idB, data->data);
    //         tdoa_data.push_back(tdoa);
    //     }
    // }

    // void transferIMUData()
    // {
    //     int i = 0;
    //     imu_data.clear();
    //     for (const auto& data : imuBuf) {
    //         if (i % 10 == 0) {
    //             Eigen::Vector3d acc(data->linear_acceleration.x, data->linear_acceleration.y, data->linear_acceleration.z);
    //             if (acc_ratio) acc *= 9.81;
    //             Eigen::Vector3d gyro(data->angular_velocity.x, data->angular_velocity.y, data->angular_velocity.z);
    //             if (gyro_unit) gyro *= M_PI / 180.0;            
    //             IMUData imu(data->header.stamp.toSec(), acc, gyro);
    //             imu_data.push_back(imu);
    //         }
    //         i++;
    //     }
    // }

    // template<typename T>
    // void slideForwardOneBuf(deque<T> &buf, double tremove)
    // {
    //     while(buf.size() != 0)
    //         if(buf.front()->header.stamp.toSec() < tremove)
    //             buf.pop_front();
    //         else
    //             break;
    // }

    // void slideForward(double tremove)
    // {
    //     if (fuse_tdoa) slideForwardOneBuf(tdoaBuf, tremove);
    //     if (fuse_tof ) slideForwardOneBuf(tofBuf,  tremove);
    //     if (fuse_imu ) slideForwardOneBuf(imuBuf,  tremove);
    // }
};

CameraImuBuf CIBuf;
// vector<Eigen::Vector3d> gtBuf;
nav_msgs::Path est_path;
nav_msgs::Path gt_path;

ros::Publisher gt_pub;
ros::Publisher gt_path_pub;
ros::Publisher est_pub;
ros::Publisher odom_pub;
ros::Publisher knot_pub;
ros::Publisher corner_pub;

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
const Eigen::Vector3d P_I_tag = Eigen::Vector3d(-0.012, 0.001, 0.091);

bool if_save_traj;
std::string traj_save_path;

void publishCornerPos(std::map<int, Eigen::Vector3d> &corner_pos_3d)
{
    sensor_msgs::PointCloud2 corners_msg;
    pcl::PointCloud<pcl::PointXYZ> pc_corners;
    for (const auto& iter : corner_pos_3d) {
        Eigen::Vector3d pos_i = iter.second;
        pc_corners.points.push_back(pcl::PointXYZ(pos_i[0], pos_i[1], pos_i[2]));
    }
    pcl::toROSMsg(pc_corners, corners_msg);
    corners_msg.header.stamp = ros::Time::now();
    corners_msg.header.frame_id = "map";
    corner_pub.publish(corners_msg);
}

void processData(GaussianProcessPtr traj, GPMVICalibPtr gpmui, std::map<int, Eigen::Vector3d> corner_pos_3d,
                CameraCalibration &cam_calib)
{
    // Loop and optimize
    while(ros::ok())
    {
        // Step 0: Check if there is data that can be admitted to the sw buffer
        double newMaxTime = CIBuf.maxTime();

//         ros::Time timeout = ros::Time::now();
//         // if(UIBuf.maxTime() < newMaxTime)
//         // {
//         //     if(auto_exit && (ros::Time::now() - timeout).toSec() > 20.0)
//         //     {
//         //         printf("Polling time out exiting.\n");
//         //         exit(-1);
//         //     }
//         //     static int msWait = int(SLIDE_SIZE*gpDt*1000);
//         //     this_thread::sleep_for(chrono::milliseconds(msWait));
//         //     continue;
//         // }
//         timeout = ros::Time::now();

//         // Step 1: Extract the data to the local buffer
//         // CIBuf.transferData(UIBuf, newMaxTime);

        // Step 2: Extend the trajectory
        if (traj->getMaxTime() < newMaxTime && (newMaxTime - traj->getMaxTime()) > gpDt*0.01) {
            traj->extendOneKnot();
        }

        // Step 3: Optimization
        TicToc tt_solve;          
        double tmin = traj->getKnotTime(0) + 1e-3;     // Start time of the sliding window
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3;      // End time of the sliding window              
        double tmid = tmin + SLIDE_SIZE*traj->getDt() + 1e-3;     // Next start time of the sliding window,
                                               // also determines the marginalization time limit          
        gpmui->Evaluate(traj, bg, ba, cam_calib, tmin, tmax, tmid, CIBuf.corner_data_cam0, CIBuf.corner_data_cam1, CIBuf.imu_data, 
                        corner_pos_3d, false, 
                        w_corner, GYR_N, ACC_N, GYR_W, ACC_W, corner_loss_thres, mp_loss_thres);
        tt_solve.Toc();

//         // Step 4: Report, visualize
//         printf("Traj: %f. Sw: %.3f -> %.3f. Buf: %d, %d, %d. Num knots: %d\n",
//                 traj->getMaxTime(), swUIBuf.minTime(), swUIBuf.maxTime(),
//                 UIBuf.tdoaBuf.size(), UIBuf.tofBuf.size(), UIBuf.imuBuf.size(), traj->getNumKnots());

//         // Visualize knots
//         pcl::PointCloud<pcl::PointXYZ> est_knots;
//         for (int i = 0; i < traj->getNumKnots(); i++) {   
//             Eigen::Vector3d knot_pos = traj->getKnotPose(i).translation();
//             est_knots.points.push_back(pcl::PointXYZ(knot_pos.x(), knot_pos.y(), knot_pos.z()));
//         }
//         sensor_msgs::PointCloud2 knot_msg;
//         pcl::toROSMsg(est_knots, knot_msg);
//         knot_msg.header.stamp = ros::Time::now();
//         knot_msg.header.frame_id = "map";        
//         knot_pub.publish(knot_msg);

//         // Visualize estimated trajectory
//         auto est_pose = traj->pose(swUIBuf.tdoa_data.front().t);
//         Eigen::Vector3d est_pos = est_pose.translation();
//         Eigen::Quaterniond est_ort = est_pose.unit_quaternion();
//         geometry_msgs::PoseStamped traj_msg;
//         traj_msg.header.stamp = ros::Time::now();
//         traj_msg.pose.position.x = est_pos.x();
//         traj_msg.pose.position.y = est_pos.y();
//         traj_msg.pose.position.z = est_pos.z();
//         traj_msg.pose.orientation.w = 1;
//         traj_msg.pose.orientation.x = 0;
//         traj_msg.pose.orientation.y = 0;
//         traj_msg.pose.orientation.z = 0;
//         est_path.poses.push_back(traj_msg);
//         est_pub.publish(est_path);

//         // Visualize odometry
//         nav_msgs::Odometry odom_msg;
//         odom_msg.header.stamp = ros::Time::now();
//         odom_msg.header.frame_id = "map";
//         est_pose = traj->pose(traj->getKnotTime(traj->getNumKnots() - 1));
//         est_pos = est_pose.translation();
//         est_ort = est_pose.unit_quaternion();
//         odom_msg.pose.pose.position.x = est_pos[0];
//         odom_msg.pose.pose.position.y = est_pos[1];
//         odom_msg.pose.pose.position.z = est_pos[2];
//         odom_msg.pose.pose.orientation.w = est_ort.w();
//         odom_msg.pose.pose.orientation.x = est_ort.x();
//         odom_msg.pose.pose.orientation.y = est_ort.y();
//         odom_msg.pose.pose.orientation.z = est_ort.z();
//         odom_pub.publish(odom_msg);             

        gt_path_pub.publish(gt_path);    
        publishCornerPos(corner_pos_3d);
        // break;

//         // // Step 5: Slide the window forward
//         // if (traj->getNumKnots() >= WINDOW_SIZE)
//         // {
//         //     double removeTime = traj->getKnotTime(traj->getNumKnots() - WINDOW_SIZE + SLIDE_SIZE);
//         //     swUIBuf.slideForward(removeTime);
//         // }
    }
    
}

void saveTraj(GaussianProcessPtr traj)
{
    if (!std::filesystem::is_directory(traj_save_path) || !std::filesystem::exists(traj_save_path)) {
        std::filesystem::create_directories(traj_save_path);
    }
    std::string traj_file_name = traj_save_path + "traj.txt";
    std::ofstream f_traj(traj_file_name);    
    for (int i = 0; i < gt_path.poses.size(); i++) {
        double t_gt = gt_path.poses[i].header.stamp.toSec();
        auto   us = traj->computeTimeIndex(t_gt);
        int    u  = us.first;
        double s  = us.second;

        if (u < 0 || u+1 >= traj->getNumKnots()) {
            continue;
        }        
        auto est_pose = traj->pose(t_gt);     
        Eigen::Vector3d est_pos = est_pose.translation();
        Eigen::Quaterniond est_ort = est_pose.unit_quaternion();    
        f_traj << std::fixed << t_gt << std::setprecision(7) 
               << " " << est_pos.x() << " " << est_pos.y() << " " << est_pos.z() 
               << " " << est_ort.x() << " " << est_ort.y() << " " << est_ort.z()  << " " << est_ort.w() << std::endl;
    }
    f_traj.close();

    std::string gt_file_name = traj_save_path + "gt.txt";
    std::ofstream f_gt(gt_file_name);    
    for (int i = 0; i < gt_path.poses.size(); i++) {
        double t_gt = gt_path.poses[i].header.stamp.toSec();
    
        f_gt << std::fixed << t_gt << std::setprecision(7) 
             << " " << gt_path.poses[i].pose.position.x << " " << gt_path.poses[i].pose.position.y << " " << gt_path.poses[i].pose.position.z
             << " " << gt_path.poses[i].pose.orientation.x << " " << gt_path.poses[i].pose.orientation.y << " " << gt_path.poses[i].pose.orientation.z  << " " << gt_path.poses[i].pose.orientation.w << std::endl;
    }
    f_gt.close();    
}

int main(int argc, char **argv)
{
    // Initialize the node
    ros::init(argc, argv, "gpvicalib");
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

    // Find the path to data
    string data_path;
    nh_ptr->getParam("data_path", data_path);
    
    // Load the corner positions in 3D and measurements
    string corner3d_path = data_path + "corners3D.csv";
    std::cout << "data_path: " << data_path << " corner3d_path: " << corner3d_path << std::endl;
    std::map<int, Eigen::Vector3d> corner_pos_3d = getCornerPosition3D(corner3d_path);

    string corner2d_path0 = data_path + "corners2D_cam0.csv";
    getCornerPosition2D(corner2d_path0, CIBuf.corner_data_cam0);
    string corner2d_path1 = data_path + "corners2D_cam1.csv";
    getCornerPosition2D(corner2d_path1, CIBuf.corner_data_cam1);    

    CameraCalibration cam_calib;

    string cam_path = data_path + "initial_calibration.json";
    getCameraModel(cam_path, cam_calib);

    getIMUMeasurements(data_path, CIBuf.imu_data);
    getGT(data_path, gt_path);

    // Topics to subscribe to
    // string tdoa_topic; nh_ptr->getParam("tdoa_topic", tdoa_topic);
    // string tof_topic;  nh_ptr->getParam("tof_topic", tof_topic);
    // string imu_topic;  nh_ptr->getParam("imu_topic", imu_topic);
    // string gt_topic;   nh_ptr->getParam("gt_topic", gt_topic);
    // fuse_tdoa = Util::GetBoolParam(nh_ptr, "fuse_tdoa", fuse_tdoa);
    // fuse_tof  = Util::GetBoolParam(nh_ptr, "fuse_tof" , fuse_tof );
    // fuse_imu  = Util::GetBoolParam(nh_ptr, "fuse_imu" , fuse_imu );

    // // Subscribe to the topics
    // tdoaSub = nh_ptr->subscribe(tdoa_topic, 10, tdoaCb);
    // tofSub  = nh_ptr->subscribe(tof_topic,  10, tofCb);
    // imuSub  = nh_ptr->subscribe(imu_topic,  10, imuCb);
    // gtSub   = nh_ptr->subscribe(gt_topic,   10, gtCb);

    // Publish estimates
    knot_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/estimated_knot", 10);
    // gt_pub   = nh_ptr->advertise<nav_msgs::Odometry>("/ground_truth", 10);
    gt_path_pub = nh_ptr->advertise<nav_msgs::Path>("/ground_truth_path", 10);
    est_pub  = nh_ptr->advertise<nav_msgs::Path>("/estimated_trajectory", 10);
    odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/estimated_pose", 10);
    corner_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/corners", 10);

    est_path.header.frame_id = "map";
    gt_path.header.frame_id = "map";

    // Time to check the buffers and perform optimization
    nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);
    nh_ptr->getParam("SLIDE_SIZE", SLIDE_SIZE);
    nh_ptr->getParam("w_corner", w_corner);
    nh_ptr->getParam("GYR_N", GYR_N);
    nh_ptr->getParam("GYR_W", GYR_W);
    nh_ptr->getParam("ACC_N", ACC_N);
    nh_ptr->getParam("ACC_W", ACC_W);    
    nh_ptr->getParam("corner_loss_thres", corner_loss_thres);
    nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
    if_save_traj = Util::GetBoolParam(nh_ptr, "if_save_traj", if_save_traj);
    nh_ptr->getParam("traj_save_path", traj_save_path);
    
    // Create the trajectory
    traj = GaussianProcessPtr(new GaussianProcess(gpDt, gpQr, gpQc, true));
    GPMVICalibPtr gpmui(new GPMVICalib(nh_ptr));

    double t0 = CIBuf.minTime();
    SE3d initial_pose;
    initial_pose.translation() = Eigen::Vector3d::Zero();
    traj->setKnot(0, GPState(t0, initial_pose));

    // Wait to get the initial time
    // while(ros::ok())
    // {
    //     if(UIBuf.minTime() == POSINF)
    //     {
    //         this_thread::sleep_for(chrono::milliseconds(5));
    //         ros::spinOnce();
    //         continue;
    //     }

    //     double t0 = UIBuf.minTime();
    //     traj->setStartTime(t0);

    //     // Set initial pose
    //     SE3d initial_pose;
    //     initial_pose.translation() = Eigen::Vector3d(1.25, 0.0, 0.07);
    //     traj->setKnot(0, GPState(t0, initial_pose));
    //     break;
    // }
    // printf(KGRN "Start time: %f\n" RESET, traj->getMinTime());

    // Start polling and processing the data
    thread pdthread(processData, traj, gpmui, corner_pos_3d, cam_calib);

    // Spin
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    pdthread.join();
    if (if_save_traj) {
        saveTraj(traj);
    }
    return 0;
}