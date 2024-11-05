#include "unistd.h"
#include <algorithm>  // for std::sort

// // PCL utilities
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/filters/uniform_sampling.h>
// #include <pcl/filters/impl/uniform_sampling.hpp>

// For json parser
#include <opencv2/opencv.hpp>

// ROS utilities
#include "ros/ros.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/PointCloud2.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "utility.h"
#include "GaussianProcess.hpp"
#include "GPVICalib.hpp"

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

        cv::FileNode intrinsic = intrinsics[i];
        cv::FileNode subintrinsic = intrinsic["intrinsics"];

        double fx, fy, cx, cy, xi, alpha;
        fx = subintrinsic["fx"];
        fy = subintrinsic["fy"];
        cx = subintrinsic["cx"];
        cy = subintrinsic["cy"];
        xi = subintrinsic["xi"];
        alpha = subintrinsic["alpha"];

        Eigen::Quaterniond qic(qw, qx, qy, qz);
        Sophus::SE3d Tic;
        Tic.translation() = Eigen::Vector3d(x, y, z);
        Tic.so3() = Sophus::SO3d::fitToSO3(qic.toRotationMatrix());
        cam_calib.T_i_c.push_back(Tic);

        DoubleSphereCamera<double> intr;
        intr.setFromInit(fx, fy, cx, cy, xi, alpha);
        cam_calib.intrinsics.push_back(intr);
        std::cout << "intr.setFromI: " << intr.getParam().transpose() << std::endl;
        std::cout << "qic: " << qic.coeffs().transpose() << " Ric: " << Tic.so3().matrix() << std::endl;
    }
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
    }
    infile.close();
    std::cout << "loaded " << gt_path.poses.size() << " gt data" << std::endl;
}

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

bool acc_ratio = false;
bool gyro_unit = false;

struct CameraImuBuf
{
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
};

CameraImuBuf CIBuf;
// nav_msgs::Path est_path;
nav_msgs::Path gt_path;

ros::Publisher gt_pub;
ros::Publisher gt_path_pub;
ros::Publisher est_pub;
ros::Publisher odom_pub;
ros::Publisher knot_pub;
ros::Publisher corner_pub;

Eigen::Vector3d bg = Eigen::Vector3d::Zero();
Eigen::Vector3d ba = Eigen::Vector3d::Zero();
Eigen::Vector3d g = Eigen::Vector3d(0, 0, 9.81);

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
                CameraCalibration* cam_calib)
{
    // Loop and optimize
    while(ros::ok())
    {
        // Step: Optimization
        TicToc tt_solve;          
        double tmin = traj->getKnotTime(0) + 1e-3;     // Start time of the sliding window
        double tmax = traj->getKnotTime(traj->getNumKnots() - 1) + 1e-3;      // End time of the sliding window              
        double tmid = tmin + SLIDE_SIZE*traj->getDt() + 1e-3;     // Next start time of the sliding window,
                                               // also determines the marginalization time limit          
        gpmui->Evaluate(traj, bg, ba, g, cam_calib, tmin, tmax, tmid, CIBuf.corner_data_cam0, CIBuf.corner_data_cam1, CIBuf.imu_data, 
                        corner_pos_3d, false, 
                        w_corner, GYR_N, ACC_N, GYR_W, ACC_W, corner_loss_thres, mp_loss_thres);
        tt_solve.Toc();

//         // Step 4: Report, visualize
//         printf("Traj: %f. Sw: %.3f -> %.3f. Buf: %d, %d, %d. Num knots: %d\n",
//                 traj->getMaxTime(), swUIBuf.minTime(), swUIBuf.maxTime(),
//                 UIBuf.tdoaBuf.size(), UIBuf.tofBuf.size(), UIBuf.imuBuf.size(), traj->getNumKnots());
        for (int i = 0; i < 2; i++) {
            std::cout << "Ric" << i  << ": " << cam_calib->T_i_c[i].so3().matrix() << std::endl;
            std::cout << "tic" << i  << ": " << cam_calib->T_i_c[i].translation().transpose() << std::endl;
            std::cout << "param" << i  << ": " << cam_calib->intrinsics[i].getParam().transpose() << std::endl;
        }
        std::cout << "ba: " << ba.transpose() << " bg: " << bg.transpose() << " g: " << g.transpose() << std::endl;
        
        // Visualize knots
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

        gt_path_pub.publish(gt_path);    
        publishCornerPos(corner_pos_3d);
        break;
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

    // Publish estimates
    knot_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/estimated_knot", 10);
    // gt_pub   = nh_ptr->advertise<nav_msgs::Odometry>("/ground_truth", 10);
    gt_path_pub = nh_ptr->advertise<nav_msgs::Path>("/ground_truth_path", 10);
    est_pub  = nh_ptr->advertise<nav_msgs::Path>("/estimated_trajectory", 10);
    odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/estimated_pose", 10);
    corner_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/corners", 10);

    // est_path.header.frame_id = "map";
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

    // double t0 = CIBuf.minTime();
    double t0 = max(CIBuf.imu_data.front().t, CIBuf.corner_data_cam0.front().t);
    traj->setStartTime(t0);
    SE3d initial_pose;
    Eigen::Matrix3d rwi;
    rwi << -0.997865,  0.0135724,  0.0638772,
            0.0628005, -0.0687564,   0.995655,
            0.0179054,   0.997541,  0.0677573;
    initial_pose.so3() = Sophus::SO3d::fitToSO3(rwi);
    initial_pose.translation() = Eigen::Vector3d(0.290213, 0.393962, 0.642399);
    traj->setKnot(0, GPState(t0, initial_pose));

    Eigen::Matrix3d rai;
    rai << -0.997945, 0.00867498, 0.0634844,
            0.0627571, -0.0675298, 0.995742,
            0.0129251, 0.99768, 0.0668466;

    for (size_t i = 0; i < CIBuf.imu_data.size(); i++) {
        const Eigen::Vector3d ad = CIBuf.imu_data[i].acc;
        if (std::abs(CIBuf.imu_data[i].t - CIBuf.corner_data_cam0[1].t) < 3000000*1e-9) {
            g = rai * ad;
            std::cout << "g_a initialized with " << g.transpose() << std::endl;
            break;
        }
    }

    double newMaxTime = min(CIBuf.imu_data.back().t, CIBuf.corner_data_cam0.back().t);

    // Step 2: Extend the trajectory
    if (traj->getMaxTime() < newMaxTime && (newMaxTime - traj->getMaxTime()) > gpDt*0.01) {
        traj->extendKnotsTo(newMaxTime, GPState(t0, initial_pose));
    }    

    // Start polling and processing the data
    thread pdthread(processData, traj, gpmui, corner_pos_3d, &cam_calib);

    // Spin
    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    pdthread.join();
    if (if_save_traj) {
        saveTraj(traj);
    }
    return 0;
}