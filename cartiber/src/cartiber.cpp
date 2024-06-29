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
#include "GaussianProcess.hpp"
#include "GPKFLO.hpp"
#include "GPMAPLO.hpp"

// Factor for optimization
// #include "factor/PoseAnalyticFactor.h"
#include "factor/ExtrinsicFactor.h"
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"
// #include "factor/ExtrinsicPoseFactor.h"
// #include "factor/GPPoseFactor.h"

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
double pmap_leaf_size = 0.15;

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

// Mutex for the node handle
mutex nh_mtx;

// Define the posespline
typedef std::shared_ptr<GPKFLO> GPKFLOPtr;
typedef std::shared_ptr<GPMAPLO> GPMAPLOPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

vector<GPKFLOPtr> gpkflo;
vector<GPMAPLOPtr> gpmaplo;

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
        pc0[lidx] = uniformDownsample<PointXYZI>(pc0[lidx], pmap_leaf_size);
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

    ceres::LocalParameterization *local_parameterization = new GPSO3dLocalParameterization();

    // Add the parameter blocks for rotation
    for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
    {
        problem.AddParameterBlock(traj->getKnotSO3(knot_idx).data(), 4, local_parameterization);
        problem.AddParameterBlock(traj->getKnotOmg(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotPos(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotVel(knot_idx).data(), 3);
        problem.AddParameterBlock(traj->getKnotAcc(knot_idx).data(), 3);
    }

    double cost_pose_init = -1;
    double cost_pose_final = -1;
    vector<ceres::internal::ResidualBlock *> res_ids_pose;
    // Add the pose factors
    // for (int k = 0; k < ts.size(); k++)
    // {
    //     double t = ts[k];

    //     // Continue if sample is in the window
    //     if (!traj->TimeInInterval(t, 1e-6))
    //         continue;

    //     auto   us = traj->computeTimeIndex(t);
    //     int    u  = us.first;
    //     double s  = us.second;

    //     Quaternd q(rot(k, 3), rot(k, 0), rot(k, 1), rot(k, 2));
    //     Vector3d p(pos(k, 0), pos(k, 1), pos(k, 2));

    //     // Find the coupled poses
    //     vector<double *> factor_param_blocks;
    //     for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
    //     {
    //         factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
    //     }

    //     double pp_loss_thres = -1.0;
    //     nh_ptr->getParam("pp_loss_thres", pp_loss_thres);
    //     ceres::LossFunction *pose_loss_function = pp_loss_thres <= 0 ? NULL : new ceres::CauchyLoss(pp_loss_thres);
    //     ceres::CostFunction *cost_function = new GPPoseFactor(SE3d(q, p), 1.0, 1.0, traj->getDt(), s);
    //     auto res_block = problem.AddResidualBlock(cost_function, pose_loss_function, factor_param_blocks);
    //     res_ids_pose.push_back(res_block);
    // }

    // Add the GP factors based on knot difference
    double cost_mp_init = -1;
    double cost_mp_final = -1;
    vector<ceres::internal::ResidualBlock *> res_ids_mp;
    // for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
    // {
    //     vector<double *> factor_param_blocks;
    //     // Add the parameter blocks
    //     for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
    //     {
    //         factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
    //         factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
    //     }

    //     // Create the factors
    //     double mp_loss_thres = -1;
    //     nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
    //     ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
    //     ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(1.0, 1.0, traj->getDt());
    //     auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
    //     res_ids_mp.push_back(res_block);
    // }

    // Init cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);

    // Final cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

    string report = myprintf("GP Fitting. Cost: %f -> %f. Iterations: %d.\n",
                             cost_pose_init, cost_pose_final, summary.iterations.size());
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

myTf<double> umeyamaAlign(const CloudPosePtr &x_, const CloudPosePtr &y_)
{
    ROS_ASSERT(x_->size() == y_->size());

    int Npoints = x_->size();
    MatrixXd x(3, Npoints);
    MatrixXd y(3, Npoints);

    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int idx = 0; idx < Npoints; idx++)
    {
        x.block<3, 1>(0, idx) << x_->points[idx].x, x_->points[idx].y, x_->points[idx].z;
        y.block<3, 1>(0, idx) << y_->points[idx].x, y_->points[idx].y, y_->points[idx].z;
    }

    return umeyamaAlign(x, y);
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
    ceres::LocalParameterization *local_parameterization = new GPSO3dLocalParameterization();

    SO3d R_Lx_Ly(Quaternd(1, 0, 0, 0));
    problem.AddParameterBlock(R_Lx_Ly.data(), 4, local_parameterization);

    Vector3d P_Lx_Ly(0, 0, 0);
    problem.AddParameterBlock(P_Lx_Ly.data(), 3);
    
    vector<double *> factor_param_blocks;
    factor_param_blocks.emplace_back(R_Lx_Ly.data());
    factor_param_blocks.emplace_back(P_Lx_Ly.data());

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

    TicToc tt_slv;

    // Init cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);

    // Final cost
    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);
    
    tt_slv.Toc();

    myTf<double> T_L0_Li(R_Lx_Ly.unit_quaternion(), P_Lx_Ly);

    printf(KCYN
           "Pose-Only Extrinsic Opt: Iter: %d. Time: %f.\n"
           "Factor: Cross: %d.\n"
           "J0: %12.3f. Xtrs: %9.3f.\n"
           "Jk: %12.3f. Xtrs: %9.3f.\n"
           "T_L0_Li. XYZ: %7.3f, %7.3f, %7.3f. YPR: %7.3f, %7.3f, %7.3f\n"
           RESET,
            summary.iterations.size(), tt_slv.GetLastStop(),
            res_ids_pose.size(),
            summary.initial_cost, cost_pose_init, 
            summary.final_cost, cost_pose_final,
            T_L0_Li.pos.x(), T_L0_Li.pos.y(), T_L0_Li.pos.z(),
            T_L0_Li.yaw(), T_L0_Li.pitch(), T_L0_Li.roll());

    return T_L0_Li;
}

myTf<double> optimizeExtrinsics(const GaussianProcessPtr &trajx, const GaussianProcessPtr &trajy)
{
    // Ceres problem
    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    // Set up the ceres problem
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = MAX_THREADS;
    options.max_num_iterations = 50;

    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *so3parameterization = new GPSO3dLocalParameterization();

    // Add params to the problem --------------------------------------------------------------------------------------

    for (int kidx = 0; kidx < trajx->getNumKnots(); kidx++)
    {
        problem.AddParameterBlock(trajx->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(trajx->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(trajx->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(trajx->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(trajx->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(trajx->getKnotAcc(kidx).data(), 3);

        // Fix the pose
        problem.SetParameterBlockConstant(trajx->getKnotSO3(kidx).data());
        problem.SetParameterBlockConstant(trajx->getKnotOmg(kidx).data());
        problem.SetParameterBlockConstant(trajx->getKnotAlp(kidx).data());
        problem.SetParameterBlockConstant(trajx->getKnotPos(kidx).data());
        problem.SetParameterBlockConstant(trajx->getKnotVel(kidx).data());
        problem.SetParameterBlockConstant(trajx->getKnotAcc(kidx).data());
    }

    for (int kidx = 0; kidx < trajy->getNumKnots(); kidx++)
    {
        problem.AddParameterBlock(trajy->getKnotSO3(kidx).data(), 4, new GPSO3dLocalParameterization());
        problem.AddParameterBlock(trajy->getKnotOmg(kidx).data(), 3);
        problem.AddParameterBlock(trajy->getKnotAlp(kidx).data(), 3);
        problem.AddParameterBlock(trajy->getKnotPos(kidx).data(), 3);
        problem.AddParameterBlock(trajy->getKnotVel(kidx).data(), 3);
        problem.AddParameterBlock(trajy->getKnotAcc(kidx).data(), 3);

        // Fix the pose
        problem.SetParameterBlockConstant(trajy->getKnotSO3(kidx).data());
        problem.SetParameterBlockConstant(trajy->getKnotOmg(kidx).data());
        problem.SetParameterBlockConstant(trajy->getKnotAlp(kidx).data());
        problem.SetParameterBlockConstant(trajy->getKnotPos(kidx).data());
        problem.SetParameterBlockConstant(trajy->getKnotVel(kidx).data());
        problem.SetParameterBlockConstant(trajy->getKnotAcc(kidx).data());
    }

    SO3d R_Lx_Ly(Quaternd(1, 0, 0, 0));
    problem.AddParameterBlock(R_Lx_Ly.data(), 4, so3parameterization);

    Vector3d P_Lx_Ly(0, 0, 0);
    problem.AddParameterBlock(P_Lx_Ly.data(), 3);

    // Fix the first pose of each trajectory
    // problem.SetParameterBlockConstant(trajx->getKnotSO3(0).data());
    // problem.SetParameterBlockConstant(trajx->getKnotPos(0).data());
    // problem.SetParameterBlockConstant(trajy->getKnotSO3(0).data());
    // problem.SetParameterBlockConstant(trajy->getKnotPos(0).data());

    //-----------------------------------------------------------------------------------------------------------------


    // Add the motion prior factors
    double cost_mp2k_init = -1;
    double cost_mp2k_final = -1;
    vector<ceres::internal::ResidualBlock *> res_ids_mp2k;
    for (int kidx = 0; kidx < trajx->getNumKnots() - 1; kidx++)
    {
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            factor_param_blocks.push_back(trajx->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(trajx->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(trajx->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(trajx->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(trajx->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(trajx->getKnotAcc(knot_idx).data());
        }

        // Create the factors
        double mpSigmaR = 1.0;
        double mpSigmaP = 1.0;
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(mpSigmaR, mpSigmaP, trajx->getDt());
        auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        res_ids_mp2k.push_back(res_block);
    }

    for (int kidx = 0; kidx < trajy->getNumKnots() - 1; kidx++)
    {
        vector<double *> factor_param_blocks;
        // Add the parameter blocks
        for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
        {
            factor_param_blocks.push_back(trajy->getKnotSO3(knot_idx).data());
            factor_param_blocks.push_back(trajy->getKnotOmg(knot_idx).data());
            factor_param_blocks.push_back(trajy->getKnotAlp(knot_idx).data());
            factor_param_blocks.push_back(trajy->getKnotPos(knot_idx).data());
            factor_param_blocks.push_back(trajy->getKnotVel(knot_idx).data());
            factor_param_blocks.push_back(trajy->getKnotAcc(knot_idx).data());
        }

        // Create the factors
        double mpSigmaR = 1.0;
        double mpSigmaP = 1.0;
        double mp_loss_thres = -1;
        // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(mpSigmaR, mpSigmaP, trajy->getDt());
        auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        res_ids_mp2k.push_back(res_block);
    }

    int Nseg = 1;

    // Add the extrinsic factors
    double cost_xtrinsic_init = -1;
    double cost_xtrinsic_final = -1;
    vector<ceres::internal::ResidualBlock *> res_ids_xtrinsic;
    for (int kidx = 0; kidx < trajx->getNumKnots() - 2; kidx++)
    {
        for(int i = 0; i < Nseg; i++)
        {
            // Get the knot time
            double t = trajx->getKnotTime(kidx) + trajx->getDt()/Nseg*i + 0*trajx->getDt()/Nseg/2;

            // Skip if time is outside the range of the other trajectory
            if (!trajy->TimeInInterval(t))
                continue;

            pair<int, double> uss, usf;
            uss = trajx->computeTimeIndex(t);
            usf = trajy->computeTimeIndex(t);

            int umins = uss.first;
            int uminf = usf.first;
            double ss = uss.second;
            double sf = usf.second;

            // Add the parameter blocks
            vector<double *> factor_param_blocks;
            for (int kidx = umins; kidx < umins + 2; kidx++)
            {
                factor_param_blocks.push_back(trajx->getKnotSO3(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotOmg(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotAlp(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotPos(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotVel(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotAcc(kidx).data());
            }
            for (int kidx = uminf; kidx < uminf + 2; kidx++)
            {
                factor_param_blocks.push_back(trajy->getKnotSO3(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotOmg(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotAlp(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotPos(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotVel(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotAcc(kidx).data());
            }
            factor_param_blocks.push_back(R_Lx_Ly.data());
            factor_param_blocks.push_back(P_Lx_Ly.data());

            // Create the factors
            double mpSigmaR = 1.0;
            double mpSigmaP = 1.0;
            double mp_loss_thres = -1;
            // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPExtrinsicFactor(mpSigmaR, mpSigmaP, trajx->getDt(), trajy->getDt(), ss, sf);
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
            res_ids_xtrinsic.push_back(res_block);
        }
    }

    for (int kidx = 0; kidx < trajy->getNumKnots() - 2; kidx++)
    {
        for(int i = 0; i < Nseg; i++)
        {
            // Get the knot time
            double t = trajy->getKnotTime(kidx) + trajy->getDt()/Nseg*i + 0*trajx->getDt()/Nseg/2;

            // Skip if time is outside the range of the other trajectory
            if (!trajx->TimeInInterval(t))
                continue;

            pair<int, double> uss, usf;
            uss = trajy->computeTimeIndex(t);
            usf = trajx->computeTimeIndex(t);

            int umins = uss.first;
            int uminf = usf.first;
            double ss = uss.second;
            double sf = usf.second;

            // Add the parameter blocks
            vector<double *> factor_param_blocks;
            for (int kidx = umins; kidx < umins + 2; kidx++)
            {
                factor_param_blocks.push_back(trajx->getKnotSO3(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotOmg(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotAlp(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotPos(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotVel(kidx).data());
                factor_param_blocks.push_back(trajx->getKnotAcc(kidx).data());
            }
            for (int kidx = uminf; kidx < uminf + 2; kidx++)
            {
                factor_param_blocks.push_back(trajy->getKnotSO3(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotOmg(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotAlp(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotPos(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotVel(kidx).data());
                factor_param_blocks.push_back(trajy->getKnotAcc(kidx).data());
            }
            factor_param_blocks.push_back(R_Lx_Ly.data());
            factor_param_blocks.push_back(P_Lx_Ly.data());

            // Create the factors
            double mpSigmaR = 1.0;
            double mpSigmaP = 1.0;
            double mp_loss_thres = -1;
            // nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPExtrinsicFactor(mpSigmaR, mpSigmaP, trajx->getDt(), trajy->getDt(), sf, ss);
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
            res_ids_xtrinsic.push_back(res_block);
        }
    }

    TicToc tt_slv;

    // Init cost
    Util::ComputeCeresCost(res_ids_mp2k, cost_mp2k_init, problem);
    Util::ComputeCeresCost(res_ids_xtrinsic, cost_xtrinsic_init, problem);

    // Solve the optimization problem
    ceres::Solve(options, &problem, &summary);

    // Final cost
    Util::ComputeCeresCost(res_ids_mp2k, cost_mp2k_final, problem);
    Util::ComputeCeresCost(res_ids_xtrinsic, cost_xtrinsic_final, problem);

    tt_slv.Toc();

    myTf<double> T_L0_Li(R_Lx_Ly.unit_quaternion(), P_Lx_Ly);

    printf(KGRN
           "GaussProc Extrinsic Opt: Iter: %d. Time: %f.\n"
           "Factor: Cross: %d, MP2K: %d.\n"
           "J0: %12.3f. Xtrs: %9.3f. MP2k: %9.3f.\n"
           "Jk: %12.3f. Xtrs: %9.3f. MP2k: %9.3f.\n"
           "T_L0_Li. XYZ: %7.3f, %7.3f, %7.3f. YPR: %7.3f, %7.3f, %7.3f\n"
           RESET,
            summary.iterations.size(), tt_slv.GetLastStop(),
            res_ids_mp2k.size(), res_ids_xtrinsic.size(),
            summary.initial_cost, cost_mp2k_init, cost_xtrinsic_init,
            summary.final_cost, cost_mp2k_final, cost_xtrinsic_final,
            T_L0_Li.pos.x(), T_L0_Li.pos.y(), T_L0_Li.pos.z(),
            T_L0_Li.yaw(), T_L0_Li.pitch(), T_L0_Li.roll());

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

        // *cloudx = *cloud1;
        
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

void ChopTheClouds(vector<CloudXYZITPtr> &clouds, int lidx)
{
    int Ncloud = clouds.size();
    int lastCloudIdx = 0;
    int lastPointIdx = -1;
    double lastCutTime = clouds.front()->points.front().t;
    double endTime = clouds.back()->points.back().t;

    while(ros::ok())
    {
        CloudXYZITPtr cloudSeg(new CloudXYZIT());
        
        // Extract all the points within lastCutTime to lastCutTime + dt
        for(int cidx = lastCloudIdx; cidx < Ncloud; cidx++)
        {
            // Shift the pointcloud base
            lastCloudIdx = cidx;
            bool segment_completed = false;
            // Check the points from the base idx
            for(int pidx = lastPointIdx + 1; pidx < clouds[cidx]->size(); pidx++)
            {
                // Update the new base
                lastPointIdx = pidx;
                
                const PointXYZIT &point = clouds[cidx]->points[pidx];
                const double &tp = point.t;

                // printf("Adding point: %d, %d. time: %f. Cuttime: %f. %f\n",
                //         cidx, cloudSeg->size(), tp, lastCutTime, lastCutTime + deltaT);
                if (tp < lastCutTime)
                {
                    if(pidx == clouds[cidx]->size() - 1)
                        lastPointIdx = -1;
                    continue;
                }

                // If point is too close, ignore
                if (Util::pointDistance(point) < 0.1)
                    continue;

                // If point is in the interval of interest, extract it
                if (lastCutTime <= tp && tp < lastCutTime + deltaT - 1e-3)
                    cloudSeg->push_back(clouds[cidx]->points[pidx]);

                // If point has exceeded the interval, exit
                if(tp >= lastCutTime + deltaT - 1e-3)
                {
                    if(pidx == clouds[cidx]->size() - 1)
                        lastPointIdx = -1;
                    segment_completed = true;
                    break;
                }
                // If we have hit the end of the cloud reset the base before moving to the next cloud
                if(pidx == clouds[cidx]->size() - 1)
                    lastPointIdx = -1;
            }

            if (segment_completed)
                break;
        }

        // Add the segment to the buffer
        if (cloudSeg->size() != 0)
        {
            // printf("GPMAPLO %d add cloud. Time: %f\n", lidx, cloudSeg->points.front().t);
            gpmaplo[lidx]->AddCloudSeg(cloudSeg);
            lastCutTime += deltaT;
        }
        else
        {
            // printf("Lidx %d. No more points.\n", lidx);
            // break;
            // this_thread::sleep_for(chrono::milliseconds(1000));
            lastCloudIdx = 0;
            lastPointIdx = -1;
        }

        if (lastCutTime > endTime)
            break;

        this_thread::sleep_for(chrono::milliseconds(int(deltaT*1000)));
    }
}

void VisualizeGndtr(CloudXYZIPtr &priormap, vector<CloudPosePtr> &gndtrCloud)
{
    // Number of lidars
    int Nlidar = gndtrCloud.size();
    
    // Create the publisher
    ros::Publisher pmpub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap_viz", 10);
    ros::Publisher gndtrPub[Nlidar];
    for(int idx = 0; idx < Nlidar; idx++)
        gndtrPub[idx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/gndtr", idx), 10);

    // Publish gndtr every 1s
    ros::Rate rate(1);
    while(ros::ok())
    {
        ros::Time currTime = ros::Time::now();

        // Publish the prior map for visualization
        Util::publishCloud(pmpub, *priormap, currTime, "world");

        // Publish the grountruth
        for(int lidx = 0; lidx < Nlidar; lidx++)
            Util::publishCloud(gndtrPub[lidx], *gndtrCloud[lidx], ros::Time::now(), "world");

        // Sleep
        rate.sleep();
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
    nh_ptr->getParam("pmap_leaf_size", pmap_leaf_size);

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
    priormap = uniformDownsample<PointXYZI>(priormap, pmap_leaf_size);
    // Create the kd tree
    printf("Building the prior map");
    kdTreeMap->setInputCloud(priormap);

    // Make the ikd-tree
    // ikdtPM = ikdtreePtr(new ikdtree(0.5, 0.6, pmap_leaf_size));

    // Converting the topic to index
    map<string, int> pctopicidx;
    for(int idx = 0; idx < pc_topics.size(); idx++)
        pctopicidx[pc_topics[idx]] = idx;

    // Storage of the pointclouds
    vector<vector<CloudXYZITPtr>> clouds(pc_topics.size());
    vector<vector<ros::Time>> cloudstamp(pc_topics.size());
    vector<tf2_msgs::TFMessage> gndtr;
    
    vector<string> queried_topics = pc_topics;
    queried_topics.push_back("/tf");

    // Load the bag file
    rosbag::Bag lidar_bag;
    lidar_bag.open(lidar_bag_file);
    rosbag::View view(lidar_bag, rosbag::TopicQuery(queried_topics));

    // Load the pointclouds
    for (rosbag::MessageInstance const m : view)
    {
        sensor_msgs::PointCloud2::ConstPtr pcMsgOuster = m.instantiate<sensor_msgs::PointCloud2>();
        livox_ros_driver::CustomMsg::ConstPtr pcMsgLivox = m.instantiate<livox_ros_driver::CustomMsg>();
        tf2_msgs::TFMessage::ConstPtr gndtrMsg = m.instantiate<tf2_msgs::TFMessage>();;

        CloudXYZITPtr cloud(new CloudXYZIT());
        ros::Time stamp;

        string topic = m.getTopic();
        int lidx = pctopicidx.find(topic) == pctopicidx.end() ? -1 : pctopicidx[topic];
        int Npoint = 0;

        // Copy the ground truth
        if (gndtrMsg != nullptr && topic == "/tf")
        {
            tf2_msgs::TFMessage msg = *gndtrMsg;
            gndtr.push_back(msg);
            continue;
        }

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

    // Extract the ground truth and publish
    vector<CloudPosePtr> gndtrCloud(Nlidar);
    for(auto &cloud : gndtrCloud)
        cloud = CloudPosePtr(new CloudPose());
    // Add points to gndtr
    for(auto &msg : gndtr)
    {
        for (auto &tf : msg.transforms)
        {
            int lidar_id = std::stoi(tf.child_frame_id.replace(0, string("lidar_").length(), string("")));
            if (lidar_id >= Nlidar)
            {
                printf(KRED "gndtr of lidar %d but it is not declared\n" RESET, lidar_id);
                continue;
            }

            // Copy the pose to the cloud
            PointPose pose;
            pose.t = tf.header.stamp.toSec();
            pose.x = tf.transform.translation.x;
            pose.y = tf.transform.translation.y;
            pose.z = tf.transform.translation.z;
            pose.qx = tf.transform.rotation.x;
            pose.qy = tf.transform.rotation.y;
            pose.qz = tf.transform.rotation.z;
            pose.qw = tf.transform.rotation.w;
            gndtrCloud[lidar_id]->push_back(pose);
        }
    }

    // Create thread for visualizing groundtruth
    thread vizGtr = thread(VisualizeGndtr, std::ref(priormap), std::ref(gndtrCloud));

    // Initial coordinates of the lidar
    vector<myTf<double>> tf_W_Li0(Nlidar);
    vector<CloudXYZIPtr> pc0(Nlidar);

    // Initialize the pose
    vector<double> timestart(Nlidar);
    getInitPose(clouds, cloudstamp, priormap, timestartup, timestart, xyzypr_W_L0, pc0, tf_W_Li0);

    static vector<ros::Publisher> pppub(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        pppub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/poseprior_%d", lidx), 1);

    // Find a preliminary trajectory for each lidar sequence
    gpkflo = vector<GPKFLOPtr>(Nlidar);
    vector<thread> findtrajkf;
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Creating the trajectory estimator
        StateWithCov Xhat0(cloudstamp[lidx].front().toSec(), tf_W_Li0[lidx].rot, tf_W_Li0[lidx].pos, Vector3d(0, 0, 0), Vector3d(0, 0, 0), 1.0);
        gpkflo[lidx] = GPKFLOPtr(new GPKFLO(lidx, Xhat0, UW_NOISE, UV_NOISE, 0.5*0.5, 0.1, nh_ptr, nh_mtx));

        // Estimate the trajectory
        findtrajkf.push_back(thread(std::bind(&GPKFLO::FindTraj, gpkflo[lidx],
                                               std::ref(kdTreeMap), std::ref(priormap),
                                               std::ref(clouds[lidx]), std::ref(cloudstamp[lidx]))));
    }

    // Wait for the trajectory estimate to finish
    for(int lidx = 0; lidx < Nlidar; lidx++)
        findtrajkf[lidx].join();

    // Split the pointcloud by time.
    vector<vector<CloudXYZITPtr>> cloudsx(Nlidar); cloudsx[0] = clouds[0];
    for(int lidx = 1; lidx < Nlidar; lidx++)
    {
        printf("Split cloud %d\n", lidx);
        syncLidar(clouds[0], clouds[lidx], cloudsx[lidx]);
    }

    // Find the trajectory with MAP optimization
    // vector<ros::Publisher> cloudsegpub(Nlidar);
    gpmaplo = vector<GPMAPLOPtr>(Nlidar);
    vector<thread> choptheclouds(Nlidar);
    vector<thread> findtrajmap(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Create the gpmaplo objects
        gpmaplo[lidx] = GPMAPLOPtr(new GPMAPLO(nh_ptr, nh_mtx, tf_W_Li0[lidx].getSE3(), lidx));
        
        // Create the cloud publisher and publisher thread
        // cloudsegpub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/cloud_segment", lidx), 1000);
        choptheclouds[lidx] = thread(ChopTheClouds, std::ref(cloudsx[lidx]), lidx);
        
        // Create the estimators
        double t0 = cloudsx[lidx].front()->points.front().t;
        findtrajmap[lidx] = thread(std::bind(&GPMAPLO::FindTraj, gpmaplo[lidx], std::ref(kdTreeMap), std::ref(priormap), t0));
    }

    // Wait for the threads to complete
    for(auto &thr : choptheclouds)
        thr.join();
    for(auto &thr : findtrajmap)
        thr.join();

    // Finish
    printf(KGRN "All GPMPLO processes finished.\n" RESET);

    // Create the pose sampling publisher
    vector<ros::Publisher> poseSamplePub(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        poseSamplePub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/pose_sampled", lidx), 1);

    // Loop in waiting
    ros::Rate rate(1);
    while(ros::ok())
    {
        // Optimize the extrinics
        for(int n = 1; n < Nlidar; n++)
        {
            GaussianProcessPtr traj0 = gpmaplo[0]->GetTraj();
            GaussianProcessPtr trajn = gpmaplo[n]->GetTraj();

            // Sample the trajectory
            double tmin = max(traj0->getMinTime(), trajn->getMinTime());
            double tmax = min(traj0->getMaxTime(), trajn->getMaxTime());

            // Sample the trajectories:
            printf("Sampling the trajectories %d, %d\n", 0, n);
            int Nsample = 1000; nh_ptr->getParam("Nsample", Nsample);
            CloudPosePtr posesample0(new CloudPose()); posesample0->resize(Nsample);
            CloudPosePtr posesamplen(new CloudPose()); posesamplen->resize(Nsample);
            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < Nsample; pidx++)
            {
                double ts = tmin + pidx*(tmax - tmin)/Nsample;
                posesample0->points[pidx] = myTf(traj0->pose(ts)).Pose6D(ts);
                posesamplen->points[pidx] = myTf(trajn->pose(ts)).Pose6D(ts);
            }

            Util::publishCloud(poseSamplePub[0], *posesample0, ros::Time::now(), "world");
            Util::publishCloud(poseSamplePub[n], *posesamplen, ros::Time::now(), "world");

            // Run the inter traj optimization
            optimizeExtrinsics(posesample0, posesamplen);
            optimizeExtrinsics(traj0, trajn);
        }

        rate.sleep();
    }
}