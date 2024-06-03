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
#include "livox_ros_driver/CustomMsg.h"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "CloudMatcher.hpp"
#include "utility.h"
#include "GPLO.hpp"
#include "GaussianProcess.hpp"
#include "SGPLO.hpp"

// Factor for optimization
#include "factor/PoseAnalyticFactor.h"
#include "factor/ExtrinsicFactor.h"
#include "factor/ExtrinsicPoseFactor.h"
#include "factor/PoseGPFactor.h"

using namespace std;

// Node handle
boost::shared_ptr<ros::NodeHandle> nh_ptr;

// Get the dense prior map
string priormap_file = "";

// Get the lidar bag file
string lidar_bag_file = "";

// Number of clouds to work with
int MAX_CLOUDS = -1;

// Get the lidar topics
vector<string> pc_topics = {"/lidar_0/points"};

// Get the lidar type
vector<string> lidar_type = {"ouster"};

// Get the prior map leaf size
double leaf_size = 0.15;

// Kdtree for priormap
KdFLANNPtr kdTreeMap(new KdFLANN());

// period to init cut off
double timestartup = 3.0;

// Spline order
int SPLINE_N = 6;

// Spline knot length
double deltaT = 0.01;

// Number of poinclouds to shift the sliding window
int sw_shift = 5;

// ikdtree of the priormap
ikdtreePtr ikdtPM;

// Noise of the angular and linear velocities
double UW_NOISE = 10.0;
double UV_NOISE = 10.0;

// Define the posespline
// using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
// typedef Matrix<double, 12, 12> MatrixNd;
typedef std::shared_ptr<GPLO> GPLOPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

template <typename PointType>
typename pcl::PointCloud<PointType>::Ptr uniformDownsample(const typename pcl::PointCloud<PointType>::Ptr &cloudin, double sampling_radius)
{
    // Downsample the pointcloud
    pcl::UniformSampling<PointType> downsampler;
    downsampler.setRadiusSearch(sampling_radius);
    downsampler.setInputCloud(cloudin);

    typename pcl::PointCloud<PointType>::Ptr cloudout(new pcl::PointCloud<PointType>());
    downsampler.filter(*cloudout);
    return cloudout;
}

void getInitPose(const vector<vector<CloudXYZITPtr>> &clouds,
                 const vector<vector<ros::Time>> &cloudstamp,
                 CloudXYZIPtr &priormap,
                 double timestartup,
                 vector<double> timestart,
                 vector<double> xyzypr_W_L0,
                 vector<CloudXYZIPtr> &pc0,
                 vector<myTf<double>> &tf_W_Li0)
{
    // Number of lidars
    int Nlidar = cloudstamp.size();

    ROS_ASSERT(pc0.size() == Nlidar);
    ROS_ASSERT(tf_W_Li0.size() == Nlidar);

    // Find the init pose of each lidar
    for (int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Merge the pointclouds in the first few seconds
        pc0[lidx] = CloudXYZIPtr(new CloudXYZI());
        int Ncloud = cloudstamp[lidx].size();
        for(int cidx = 0; cidx < Ncloud; cidx++)
        {
            // Check if pointcloud is later
            if ((cloudstamp[lidx][cidx] - cloudstamp[lidx][0]).toSec() > timestartup)
            {
                timestart[lidx] = cloudstamp[lidx][cidx].toSec();
                break;
            }

            // Merge lidar
            CloudXYZI temp; pcl::copyPointCloud(*clouds[lidx][cidx], temp);
            *pc0[lidx] += temp;

            // printf("P0 lidar %d, Cloud %d. Points: %d. Copied: %d\n", lidx, cidx, clouds[lidx][cidx]->size(), pc0[lidx]->size());
        }

        int Norg = pc0[lidx]->size();

        // Downsample the pointcloud
        pc0[lidx] = uniformDownsample<PointXYZI>(pc0[lidx], leaf_size*2);
        printf("P0 lidar %d, Points: %d -> %d\n", lidx, Norg, pc0[lidx]->size());

        // Find ICP alignment and refine
        CloudMatcher cm(0.1, 0.1);

        // Set the original position of the anchors
        Vector3d p_W_L0(xyzypr_W_L0[lidx*6 + 0], xyzypr_W_L0[lidx*6 + 1], xyzypr_W_L0[lidx*6 + 2]);
        Quaternd q_W_L0 = Util::YPR2Quat(xyzypr_W_L0[lidx*6 + 3], xyzypr_W_L0[lidx*6 + 4], xyzypr_W_L0[lidx*6 + 5]);
        myTf tf_W_L0(q_W_L0, p_W_L0);

        // Find ICP pose
        Matrix4f tfm_W_Li0;
        double   icpFitness   = 0;
        double   icpTime      = 0;
        bool     icpconverged = cm.CheckICP(priormap, pc0[lidx], tf_W_L0.cast<float>().tfMat(), tfm_W_Li0, 0.2, 10, 1.0, icpFitness, icpTime);
        
        tf_W_L0 = myTf(tfm_W_Li0);
        printf("Lidar %d initial pose. %s. Time: %f. Fn: %f. XYZ: %f, %f, %f. YPR: %f, %f, %f.\n",
                lidx, icpconverged ? "Conv" : "Not Conv", icpTime, icpFitness,
                tf_W_L0.pos.x(), tf_W_L0.pos.y(), tf_W_L0.pos.z(),
                tf_W_L0.yaw(), tf_W_L0.pitch(), tf_W_L0.roll());

        // Find the refined pose
        IOAOptions ioaOpt;
        ioaOpt.init_tf = myTf(tfm_W_Li0);
        ioaOpt.max_iterations = 20;
        ioaOpt.show_report = true;
        ioaOpt.text = myprintf("T_W_L(%d,0)_refined_%d", lidx, 10);
        IOASummary ioaSum;
        ioaSum.final_tf = ioaOpt.init_tf;
        cm.IterateAssociateOptimize(ioaOpt, ioaSum, priormap, pc0[lidx]);
        printf("Refined: \n");
        cout << ioaSum.final_tf.tfMat() << endl;

        // Save the result to external buffer
        tf_W_Li0[lidx] = ioaSum.final_tf;
        
        // Transform pointcloud to world frame
        // pcl::transformPointCloud(*pc0[lidx], *pc0[lidx], ioaSum.final_tf.cast<float>().tfMat());
    }

    // // Return the result
    // return T_W_Li0;
}

// void deskewBySpline(CloudXYZITPtr &cloudin, CloudXYZITPtr &cloudout, PoseSplinePtr &traj)
// {
//     int Npoints = cloudin->size();
//     cloudout->resize(Npoints);
    
//     // double t0 = cloudin->points[0].t;
//     // myTf T_W_Bk(traj->pose(t0));

//     #pragma omp parallel for num_threads(MAX_THREADS)
//     for(int pidx = 0; pidx < Npoints; pidx++)
//     {        
//         PointXYZIT &pi = cloudin->points[pidx];
//         PointXYZIT &po = cloudout->points[pidx];
        
//         double ts = pi.t;
//         myTf T_W_Bs = traj->pose(ts);

//         Vector3d pi_inBs(pi.x, pi.y, pi.z);
//         Vector3d pi_inW = T_W_Bs.rot*pi_inBs + T_W_Bs.pos;

//         po.x = pi_inW.x();
//         po.y = pi_inW.y();
//         po.z = pi_inW.z();
//         po.t = pi.t;
//         po.intensity = pi.intensity;
//     }
// }

// Fit the spline with data
string FitGP(GaussianProcessPtr &traj, vector<double> ts, MatrixXd pos, MatrixXd rot, vector<double> wp, vector<double> wr, double loss_thres)
{
    // Create spline
    int KNOTS = traj->getNumKnots();

    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;

    ceres::LossFunction *loss_function = new ceres::CauchyLoss(loss_thres);
    ceres::LocalParameterization *local_parameterization = new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>();

    // Add the parameter blocks for rotation
    for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
    {
        problem.AddParameterBlock(traj->getKnotSO3(knot_idx).data(), 4, local_parameterization);
        problem.AddParameterBlock(traj->getKnotOmg(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotPos(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotVel(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAcc(knot_idx).data(), 3);
    }

    double cost_pose_init;
    double cost_pose_final;
    vector<ceres::internal::ResidualBlock *> res_ids_pose;
    // Add the pose factors
    for (int k = 0; k < ts.size(); k++)
    {
        double t = ts[k];

        // Continue if sample is in the window
        if (t < traj->getMinTime() + 1e-6 && t > traj->getMaxTime() - 1e-6)
            continue;

        auto   us = traj->computeTimeIndex(t);
        int    u  = us.first;
        double s  = us.second;

        Quaternd q(rot(k, 3), rot(k, 0), rot(k, 1), rot(k, 2));
        Vector3d p(pos(k, 0), pos(k, 1), pos(k, 2));

        ceres::CostFunction *cost_function = new PoseGPFactor(SE3d(q, p), wp[k], wr[k], traj->getDt(), s);

        // Find the coupled poses
        vector<double *> factor_param_blocks;
        for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
        {
            factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        }

        // printf("Creating functor: u: %f, s: %d. sample: %d / %d\n", u, s, i, pose_gt.size());
        // cost_function->SetNumResiduals(6);
        auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
        res_ids_pose.push_back(res_block);
    }

    // Init cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);

    // Final cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

    string report = myprintf("Spline Fitting. Cost: %f -> %f. Iterations: %d.\n",
                             cost_pose_init, cost_pose_final, summary.iterations.size());
    cout << report << endl;
    return report;
}

myTf<double> umeyamaAlign(const MatrixXd &x, const MatrixXd &y)
{
    int Nsample = x.cols();

    // Calculate the mean
    Vector3d meanx(0, 0, 0);
    Vector3d meany(0, 0, 0);
    for(int sidx = 0; sidx < Nsample; sidx++)
    {
        meanx += x.col(sidx);
        meany += y.col(sidx);
    }
    meanx /= Nsample;
    meany /= Nsample;

    // variance
    // double sigmax = 0;
    // double sigmay = 0;
    MatrixXd covxy = MatrixXd::Zero(3, 3);
    for(int sidx = 0; sidx < Nsample; sidx++)
    {
        Vector3d xtilde = x.col(sidx) - meanx;
        Vector3d ytilde = y.col(sidx) - meany;
        // sigmax += pow(xtilde.norm(), 2)/n;
        // sigmay += pow(ytilde.norm(), 2)/n;
        covxy += xtilde*ytilde.transpose();
    }
    covxy /= Nsample;

    // SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covxy, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Matrix3d U = svd.matrixU();
    // VectorXd S = svd.singularValues();
    Matrix3d V = svd.matrixV();

    double detU = U.determinant();
    double detV = V.determinant();

    Matrix3d S = MatrixXd::Identity(3, 3);
    if (detU * detV < 0.0)
        S(2, 2) = -1;
    
    Matrix3d R = U*S*V;
    Vector3d p = meany - R*meanx;

    return myTf(R, p);
}

myTf<double> optimizeExtrinsics(const CloudPosePtr &trajx, const CloudPosePtr &trajy)
{
    // Trajectory should have 1:1 correspondence
    ROS_ASSERT(trajx->size() == trajy->size());
    int Nsample = trajx->size();

    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;

    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>();

    SO3d rot(Quaternd(1, 0, 0, 0));
    problem.AddParameterBlock(rot.data(), 4, local_parameterization);

    Vector3d pos(0, 0, 0);
    problem.AddParameterBlock(pos.data(), 3);
    
    vector<double *> factor_param_blocks;
    factor_param_blocks.emplace_back(rot.data());
    factor_param_blocks.emplace_back(pos.data());

    double cost_pose_init;
    double cost_pose_final;
    vector<ceres::internal::ResidualBlock *> res_ids_pose;
    for(int sidx = 0; sidx < Nsample; sidx++)
    {
        myTf Ti(trajx->points[sidx]);
        myTf Tj(trajy->points[sidx]);
        SE3d Tji = (Ti.inverse()*Tj).getSE3();

        ceres::CostFunction *cost_function = new ExtrinsicFactor(Tji.so3(), Tji.translation(), 1.0, 1.0);
        auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
        res_ids_pose.push_back(res_block);
    }

    // Init cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);

    // Final cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

    printf("Optimization done: Iter: %d. Cost: %f -> %f\n",
            summary.iterations.size(),
            summary.initial_cost, summary.final_cost);

    myTf<double> T_L0_Li(rot.unit_quaternion(), pos);

    // Delete the created memories
    // delete rot;
    // delete pos;

    return T_L0_Li;
}

void syncLidar(const vector<CloudXYZITPtr> &cloudbuf1, const vector<CloudXYZITPtr> &cloudbuf2, vector<CloudXYZITPtr> &cloud21)
{
    int Ncloud1 = cloudbuf1.size();
    int Ncloud2 = cloudbuf2.size();

    int last_cloud2 = 0;
    for(int cidx1 = 0; cidx1 < Ncloud1; cidx1++)
    {
        // Create a cloud
        cloud21.push_back(CloudXYZITPtr(new CloudXYZIT()));

        const CloudXYZITPtr &cloud1 = cloudbuf1[cidx1];
        CloudXYZITPtr &cloudx = cloud21.back();

        *cloudx = *cloud1;
        
        for(int cidx2 = last_cloud2; cidx2 < Ncloud2; cidx2++)
        {
            const CloudXYZITPtr &cloud2 = cloudbuf2[cidx2];

            double t1f = cloud1->points.front().t;
            double t1b = cloud1->points.back().t;
            double t2f = cloud2->points.front().t;
            double t2b = cloud2->points.back().t;
            
            if (t2f > t1b)
                break;
            
            if (t2b < t1f)
                continue;

            // Now there is overlap, extract the points in the cloud1 interval
            ROS_ASSERT((t2f >= t1f && t2f <= t1b) || (t2b >= t1f && t2b <= t1b));
            // Insert points to the cloudx
            for(auto &point : cloud2->points)
                if(point.t >= t1f && point.t <= t1b)
                    cloudx->push_back(point);
            last_cloud2 = cidx2;

            // printf("Cloud2 %d is split. Cloudx of Cloud1 %d now has %d points\n", last_cloud2, cidx1, cloudx->size());
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cartiber");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
    
    // Supress the pcl warning
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Suppress warnings by pcl load

    printf("Lidar calibration started.\n");

    // GaussianProcess gp(0.05);
    // gp.setStartTime(43.91);
    // gp.extendKnotsTo(57.1102, StateStamped());

    // printf("gp: %f. %f -> %f. Knots: %d\n", gp.getDt(), gp.getMinTime(), gp.getMaxTime(), gp.getNumKnots());

    // StateStamped Xhat = gp.getStateAt(49.69);

    // cout << "Xhat.t\n" << Xhat.t << endl;
    // cout << "Xhat.R\n" << Xhat.R.matrix() << endl;
    // cout << "Xhat.O\n" << Xhat.O << endl;
    // cout << "Xhat.P\n" << Xhat.P << endl;
    // cout << "Xhat.V\n" << Xhat.V << endl;
    // cout << "Xhat.A\n" << Xhat.A << endl;

    // Spline order
    nh_ptr->getParam("SPLINE_N", SPLINE_N);
    nh_ptr->getParam("deltaT", deltaT);

    printf("SPLINE order %d with knot length: %f\n", SPLINE_N, deltaT);

    // Get the user define parameters
    nh_ptr->getParam("priormap_file", priormap_file);
    nh_ptr->getParam("lidar_bag_file", lidar_bag_file);
    nh_ptr->getParam("MAX_CLOUDS", MAX_CLOUDS);
    nh_ptr->getParam("pc_topics",  pc_topics);
    nh_ptr->getParam("lidar_type", lidar_type);
    
    printf("Get bag at %s and prior map at %s\n", lidar_bag_file.c_str(), priormap_file.c_str());
    printf("MAX_CLOUDS: %d\n", MAX_CLOUDS);

    printf("Lidar topics: \n");
    for(auto topic : pc_topics)
        cout << topic << endl;
    
    printf("Lidar type: \n");
    for(auto type : lidar_type)
        cout << type << endl;


    // Get the leaf size
    nh_ptr->getParam("leaf_size", leaf_size);

    // Noise
    nh_ptr->getParam("UW_NOISE", UW_NOISE);
    nh_ptr->getParam("UV_NOISE", UV_NOISE);
    printf("Proccess noise: %.3f, %.3f\n", UW_NOISE, UV_NOISE);

    // Determine the number of lidar
    int Nlidar = pc_topics.size();

    // Get the initial position of the lidars
    vector<double> xyzypr_W_L0(Nlidar*6, 0.0);
    if( nh_ptr->getParam("xyzypr_W_L0", xyzypr_W_L0) )
    {
        if (xyzypr_W_L0.size() < Nlidar*6)
        {
            printf(KYEL "T_W_L0 missing values. Setting all to zeros \n" RESET);
            xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            printf("T_W_L0 found: \n");
            for(int i = 0; i < Nlidar; i++)
                for(int j = 0; j < 6; j++)
                    printf("%f, ", xyzypr_W_L0[i*6 + j]);
                cout << endl;
        }
    }
    else
    {
        printf("Failed to get xyzypr_W_L0. Setting all to zeros\n");
        xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
    }

    // Load the priormap
    CloudXYZIPtr priormap(new CloudXYZI()); pcl::io::loadPCDFile<PointXYZI>(priormap_file, *priormap);
    priormap = uniformDownsample<PointXYZI>(priormap, leaf_size);
    // Create the kd tree
    printf("Building the prior map");
    kdTreeMap->setInputCloud(priormap);
    
    // Publish the prior map for visualization
    static ros::Publisher pmpub = nh.advertise<sensor_msgs::PointCloud2>("/priormap_viz", 10);
    for(int count = 0; count < 6; count ++)
    {
        Util::publishCloud(pmpub, *priormap, ros::Time::now(), "world");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Make the ikd-tree
    // ikdtPM = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));

    // Converting the topic to index
    map<string, int> pctopicidx; for(int idx = 0; idx < pc_topics.size(); idx++) pctopicidx[pc_topics[idx]] = idx;

    // Storage of the pointclouds
    vector<vector<CloudXYZITPtr>> clouds(pc_topics.size());
    vector<vector<ros::Time>> cloudstamp(pc_topics.size());

    // Load the bag file
    rosbag::Bag lidar_bag;
    lidar_bag.open(lidar_bag_file);
    rosbag::View view(lidar_bag, rosbag::TopicQuery(pc_topics));

    // Load the pointclouds
    for (rosbag::MessageInstance const m : view)
    {
        sensor_msgs::PointCloud2::ConstPtr pcMsgOuster = m.instantiate<sensor_msgs::PointCloud2>();
        livox_ros_driver::CustomMsg::ConstPtr pcMsgLivox = m.instantiate<livox_ros_driver::CustomMsg>();

        CloudXYZITPtr cloud(new CloudXYZIT());
        ros::Time stamp;
        int lidx = pctopicidx[m.getTopic()];
        int Npoint = 0;

        if (pcMsgOuster != nullptr && lidar_type[lidx] == "ouster")
        {
            Npoint = pcMsgOuster->width*pcMsgOuster->height;

            cloud->resize(Npoint);
            stamp = pcMsgOuster->header.stamp;

            CloudOusterPtr cloud_raw(new CloudOuster());
            pcl::fromROSMsg(*pcMsgOuster, *cloud_raw);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < Npoint; pidx++)
            {
                double pt0 = pcMsgOuster->header.stamp.toSec();
                PointOuster &pi = cloud_raw->points[pidx];
                PointXYZIT &po = cloud->points[pidx];
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.intensity = pi.intensity;
                po.t = pt0 + pi.t/1.0e9;
            }
        }
        else if (pcMsgLivox != nullptr && lidar_type[lidx] == "livox")
        {
            Npoint = pcMsgLivox->point_num;

            cloud->resize(Npoint);
            stamp = pcMsgLivox->header.stamp;

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < Npoint; pidx++)
            {
                double pt0 = pcMsgLivox->header.stamp.toSec();
                const livox_ros_driver::CustomPoint &pi = pcMsgLivox->points[pidx];
                PointXYZIT &po = cloud->points[pidx];
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.intensity = pi.reflectivity/255.0*1000;
                po.t = pt0 + pi.offset_time/1.0e9;
            }
        }

        if (cloud->size() != 0)
        {
            clouds[lidx].push_back(cloud);
            cloudstamp[lidx].push_back(stamp);

            printf("Loading pointcloud from lidar %d at time: %.3f, %.3f. Cloud total: %d. Cloud size: %d / %d. Topic: %s.\r",
                    lidx,
                    cloudstamp[lidx].back().toSec(),
                    clouds[lidx].back()->points.front().t,
                    clouds[lidx].size(), clouds[lidx].back()->size(), Npoint, pc_topics[lidx].c_str());
            cout << endl;

            // Confirm the time correctness
            ROS_ASSERT_MSG(cloudstamp[lidx].back().toSec() == clouds[lidx].back()->points.front().t,
                           "Time: %f, %f.",
                           cloudstamp[lidx].back().toSec(), clouds[lidx].back()->points.front().t);
        }

        if (MAX_CLOUDS > 0 && clouds.front().size() >= MAX_CLOUDS)
            break;
    }

    vector<myTf<double>> tf_W_Li0(Nlidar);
    vector<CloudXYZIPtr> pc0(Nlidar);

    // Initialize the pose
    vector<double> timestart(Nlidar);
    getInitPose(clouds, cloudstamp, priormap, timestartup, timestart, xyzypr_W_L0, pc0, tf_W_Li0);

    static vector<ros::Publisher> pppub(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        pppub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/poseprior_%d", lidx), 1);

    // Find a preliminary trajectory for each lidar sequence
    mutex nh_mtx;
    vector<GPLOPtr> gplo;
    vector<thread> trajEst;
    vector<CloudPosePtr> posePrior(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Creating the trajectory estimator
        StateWithCov Xhat0(cloudstamp[lidx].front().toSec(), tf_W_Li0[lidx].rot, tf_W_Li0[lidx].pos, Vector3d(0, 0, 0), Vector3d(0, 0, 0), 1.0);
        gplo.push_back(GPLOPtr(new GPLO(lidx, Xhat0, UW_NOISE, UV_NOISE, 0.5*0.5, 0.1, nh_ptr, nh_mtx)));

        // Estimate the trajectory
        posePrior[lidx] = CloudPosePtr(new CloudPose());
        trajEst.push_back(thread(std::bind(&GPLO::FindTraj, gplo[lidx],
                                            std::ref(kdTreeMap), std::ref(priormap),
                                            std::ref(clouds[lidx]), std::ref(cloudstamp[lidx]),
                                            std::ref(posePrior[lidx]))));
    }
    // Wait for the trajectory estimate to finish
    for(int lidx = 0; lidx < Nlidar; lidx++)
        trajEst[lidx].join();

    trajEst.clear();
    gplo.clear();

    // Merge the time stamps
    double tmincut = cloudstamp.front().front().toSec();
    double tmaxcut = cloudstamp.front().back().toSec();
    deque<double> tsample;
    for(auto &stamps : cloudstamp)
    {
        tmincut = max(tmincut, stamps.front().toSec());
        tmaxcut = min(tmaxcut, stamps.back().toSec());

        for(auto &stamp : stamps)
            tsample.push_back(stamp.toSec());
    }
    // Sort the sampling time:
    std::sort(tsample.begin(), tsample.end());
    // Pop the ones outside of tmincut and tmaxcut
    while(tsample.size() != 0)
        if (tsample.front() <= tmincut)
            tsample.pop_front();
        else
            break;

    while(tsample.size() != 0)
        if (tsample.back() >= tmaxcut)
            tsample.pop_back();
        else
            break;

    printf("Sample time: %f -> %f. %f, %f, %f, %f\n",
            tsample.front(), tsample.back(),
            cloudstamp.front().front().toSec(), cloudstamp.front().back().toSec(),
            cloudstamp.back().front().toSec(), cloudstamp.back().back().toSec());

    // Now got all trajectories, fit a spline to these trajectories and sample them
    vector<GaussianProcessPtr> traj(Nlidar);
    vector<CloudPosePtr> poseSampled(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        traj[lidx] = GaussianProcessPtr(new GaussianProcess(deltaT));
        traj[lidx]->setStartTime(clouds[lidx].front()->points.front().t);
        traj[lidx]->extendKnotsTo(clouds[lidx].back()->points.back().t);
        
        int Ncloud = clouds[lidx].size();
        vector<double> ts(Ncloud);
        MatrixXd pos(Ncloud, 3);
        MatrixXd rot(Ncloud, 4);
        vector<double> wp(Ncloud, 1);
        vector<double> wr(Ncloud, 1);
        double loss_thread = 1.0;

        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int cidx = 0; cidx < Ncloud; cidx++)
        {
            ts[cidx] = clouds[lidx][cidx]->points.back().t;
            pos.block<1, 3>(cidx, 0) << posePrior[lidx]->points[cidx].x,  posePrior[lidx]->points[cidx].y,  posePrior[lidx]->points[cidx].z;
            rot.block<1, 4>(cidx, 0) << posePrior[lidx]->points[cidx].qx, posePrior[lidx]->points[cidx].qy, posePrior[lidx]->points[cidx].qz, posePrior[lidx]->points[cidx].qw;
        }
        
        // Fit the spline
        string report = FitGP(traj[lidx], ts, pos, rot, wp, wr, loss_thread);

        // Sample the spline by the synchronized sampling time
        poseSampled[lidx] = CloudPosePtr(new CloudPose());
        for(int tidx = 0; tidx < tsample.size() - 1; tidx++)
        {
            double t1 = tsample[tidx];
            double t2 = (tsample[tidx] + tsample[tidx + 1])/2;

            poseSampled[lidx]->points.push_back(myTf(traj[lidx]->pose(t1)).Pose6D(t1));
            poseSampled[lidx]->points.push_back(myTf(traj[lidx]->pose(t2)).Pose6D(t2));
        }
    }

    // Export the sampling time for checking
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        poseSampled[lidx]->width = 1;
        poseSampled[lidx]->height = poseSampled[lidx]->size();
        printf("LIDX %d. WIDTH %d. HEIGHT: %d. Size: %d\n", lidx, poseSampled[lidx]->width, poseSampled[lidx]->height, poseSampled[lidx]->size());
        pcl::io::savePCDFileASCII (myprintf("/home/tmn/ros_ws/dev_ws/src/cartiber/log/sampled_%d.pcd", lidx), *poseSampled[lidx]);
    }
    
    // Optimize the extrinsics for initial condition
    vector<SE3d> T_L0_Li(Nlidar, SE3d());
    for(int lidx = 1; lidx < Nlidar; lidx++)
    {
        myTf tf_L0_L = optimizeExtrinsics(poseSampled[0], poseSampled[lidx]);
        printf("T_L0_L%d: XYZ: %f, %f, %f, YPR: %f, %f, %f\n",
                lidx, tf_L0_L.pos(0), tf_L0_L.pos(1),  tf_L0_L.pos(2),
                      tf_L0_L.yaw(),  tf_L0_L.pitch(), tf_L0_L.roll());
        
        // Import the estimte
        T_L0_Li[lidx] = tf_L0_L.getSE3();
    }

    // Merge the pointcloud by time.
    vector<vector<CloudXYZITPtr>> cloudsx(Nlidar); cloudsx[0] = clouds[0];
    for(int lidx = 1; lidx < Nlidar; lidx++)
    {
        printf("Split cloud %d\n", lidx);
        syncLidar(clouds[0], clouds[lidx], cloudsx[lidx]);
    }

    // // Find the trajectory with joint factors
    // SGPLO sgplo(nh_ptr, traj, T_L0_Li);

    // // Find the trajectories
    // sgplo.FindTraj(kdTreeMap, priormap, clouds);

    ros::Rate rate(1);
    while(ros::ok())
    {
        ros::Time currTime = ros::Time::now();

        // Publish the prior map for visualization
        Util::publishCloud(pmpub, *priormap, currTime, "world");

        static vector<ros::Publisher> splineSamplePub;
        if(splineSamplePub.size() == 0)
            for(int lidx = 0; lidx < Nlidar; lidx++)
                splineSamplePub.push_back(nh.advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/spline_sample", lidx), 1));

        for(int lidx = 0; lidx < Nlidar; lidx++)
            Util::publishCloud(splineSamplePub[lidx], *poseSampled[lidx], ros::Time::now(), "world");

        // Sleep
        rate.sleep();
    }
}