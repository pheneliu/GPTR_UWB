#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utility.h"

/* All needed for filter of custom point type----------*/
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/impl/crop_box.hpp>
/* All needed for filter of custom point type----------*/

#include "GaussianProcess.hpp"
#include "factor/GPPoseFactor.h"
#include "factor/GPMotionPriorFactor.h"
#include "factor/GPPointToPlaneFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"
#include "factor/GPSmoothnessFactor.h"

#include "factor/GPPoseFactorAutodiff.h"
#include "factor/GPMotionPriorFactorAutodiff.h"
#include "factor/GPPointToPlaneFactorAutodiff.h"
#include "factor/GPMotionPriorTwoKnotsFactorAutodiff.h"

#include "GNSolver.h"

// #include "basalt/spline/se3_spline.h"
// #include "basalt/spline/ceres_spline_helper.h"
// #include "basalt/spline/ceres_local_param.hpp"

using NodeHandlePtr = boost::shared_ptr<ros::NodeHandle>;
// using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

struct FactorMeta
{
    vector<double *> so3_parameter_blocks;
    vector<double *> r3_parameter_blocks;
    vector<ceres::ResidualBlockId> residual_blocks;

    int parameter_blocks()
    {
        return (so3_parameter_blocks.size() + r3_parameter_blocks.size());
    }
};

Eigen::MatrixXd GetJacobian(ceres::CRSMatrix &J)
{
    Eigen::MatrixXd dense_jacobian(J.num_rows, J.num_cols);
    dense_jacobian.setZero();
    for (int r = 0; r < J.num_rows; ++r)
    {
        for (int idx = J.rows[r]; idx < J.rows[r + 1]; ++idx)
        {
            const int c = J.cols[idx];
            dense_jacobian(r, c) = J.values[idx];
        }
    }

    return dense_jacobian;
}

void GetFactorJacobian(ceres::Problem &problem, FactorMeta &factorMeta,
                       int local_pamaterization_type,
                       double &cost, vector<double> &residual,
                       MatrixXd &Jacobian)
{
    ceres::LocalParameterization *localparameterization;
    for(auto parameter : factorMeta.so3_parameter_blocks)
    {
        if (local_pamaterization_type == 0)
        {
            localparameterization = new basalt::LieLocalParameterization<SO3d>();
            problem.SetParameterization(parameter, localparameterization);
        }
        else
        {   
            localparameterization = new basalt::LieAnalyticLocalParameterization<SO3d>();
            problem.SetParameterization(parameter, localparameterization);
        }
    }

    ceres::Problem::EvaluateOptions e_option;
    ceres::CRSMatrix Jacobian_;
    e_option.residual_blocks = factorMeta.residual_blocks;
    problem.Evaluate(e_option, &cost, &residual, NULL, &Jacobian_);
    Jacobian = GetJacobian(Jacobian_);
}

void RemoveResidualBlock(ceres::Problem &problem, FactorMeta &factorMeta)
{
    for(auto res_block : factorMeta.residual_blocks)
        problem.RemoveResidualBlock(res_block);
}

class GPMAPLO
{
private:

    NodeHandlePtr nh_ptr;

    // Debug param
    double SKIPPED_TIME;
    
    // Index for distinguishing between clouds
    int LIDX;

    // Leaf size to downsample input pointcloud
    double ds_size;

    // How many point clouds to import into the sliding window
    int WINDOW_SIZE = 10;

    // Switch to use ceres
    bool use_ceres = false;

    // My custom Gauss Newton solver
    GNSolver *mySolver;

    // Outer iterations
    int max_gniter = 3;

    // Initial pose of the lidars
    SE3d T_W_Li0;

    // Extrinsics of the lidars
    vector<SE3d> T_L0_Li;

    // Spline to model the trajectory of each lidar
    vector<GaussianProcessPtr> traj;

    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;
    double minKnnNbrDis = 0.1;

    // Rate of skipping the factors
    int lidar_ds_rate = 100;

    int SPLINE_N = 4;
    double deltaT = 0.1;

    int DK = 1;
    double tshift = 0.05;

    double lidar_weight = 1.0;
    double ppSigmaR = 10;
    double ppSigmaP = 10;
    double mpSigmaR = 10;
    double mpSigmaP = 10;
    double smSigmaR = 10;
    double smSigmaP = 10;

    // Time to fix the knot
    double fixed_start = 0.0;
    double fixed_end = 0.0;

    // Buffer for the pointcloud segments
    mutex cloud_seg_buf_mtx;
    deque<CloudXYZITPtr> cloud_seg_buf;

public:

    // Destructor
   ~GPMAPLO() {};

    GPMAPLO(NodeHandlePtr &nh_ptr_, const SE3d &T_W_Li0_, int &LIDX_)
        : nh_ptr(nh_ptr_), T_W_Li0(T_W_Li0_), LIDX(LIDX_), mySolver(new GNSolver(nh_ptr))
    {
        // Time to skip before doing the MAP optimization
        nh_ptr->getParam("SKIPPED_TIME", SKIPPED_TIME);

        // Trajectory estimate
        nh_ptr->getParam("SPLINE_N", SPLINE_N);
        nh_ptr->getParam("deltaT", deltaT);

        // Leaf size to downsample the pontclouds
        nh_ptr->getParam("ds_size", ds_size);

        // Window size
        nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);
        use_ceres = Util::GetBoolParam(nh_ptr, "use_ceres", false);

        // GN solver param
        nh_ptr->getParam("max_gniter", max_gniter);

        // Fixed points on the trajectory
        nh_ptr->getParam("fixed_start", fixed_start);
        nh_ptr->getParam("fixed_end", fixed_end);

        // Parameter for motion prior selection
        nh_ptr->getParam("DK", DK);
        nh_ptr->getParam("tshift", tshift);

        // How many points are skipped before one lidar factor is built
        nh_ptr->getParam("lidar_ds_rate", lidar_ds_rate);

        // Weight for the lidar factor
        nh_ptr->getParam("lidar_weight", lidar_weight);
        // Weight for the pose priors
        nh_ptr->getParam("ppSigmaR", ppSigmaR);
        nh_ptr->getParam("ppSigmaP", ppSigmaP);
        // Weight for the motion prior
        nh_ptr->getParam("mpSigmaR", mpSigmaR);
        nh_ptr->getParam("mpSigmaP", mpSigmaP);
        // Weight for the smoothness factor
        nh_ptr->getParam("smSigmaR", smSigmaR);
        nh_ptr->getParam("smSigmaP", smSigmaP);

        printf("Window size: %d. Fixes: %f, %f. DK: %f, %d. lidar_weight: %f. ppSigma: %f, %f. mpSigmaR: %f, %f\n\n",
                WINDOW_SIZE, fixed_start, fixed_end, tshift, DK, lidar_weight, ppSigmaR, ppSigmaP, mpSigmaR, mpSigmaP);
    }

    void Associate(const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZITPtr &cloudRaw, const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        ROS_ASSERT(cloudRaw->size() == cloudInB->size());

        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                double tpoint = cloudRaw->points[pidx].t;
                PointXYZIT pointRaw = cloudRaw->points[pidx];
                PointXYZI  pointInB = cloudInB->points[pidx];
                PointXYZI  pointInW = cloudInW->points[pidx];

                Coef_[pidx].n = Vector4d(0, 0, 0, 0);
                Coef_[pidx].t = -1;

                if(!Util::PointIsValid(pointInB))
                {
                    // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                    pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                    continue;
                }

                if(!Util::PointIsValid(pointInW))
                    continue;

                vector<int> knn_idx(knnSize, 0); vector<float> knn_sq_dis(knnSize, 0);
                kdtreeMap->nearestKSearch(pointInW, knnSize, knn_idx, knn_sq_dis);

                vector<PointXYZI> nbrPoints;
                if (knn_sq_dis.back() < minKnnSqDis)
                    for(auto &idx : knn_idx)
                        nbrPoints.push_back(priormap->points[idx]);
                else
                    continue;

                // Fit the plane
                if(Util::fitPlane(nbrPoints, 0.5, 0.1, Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    ROS_ASSERT(tpoint >= 0);
                    Coef_[pidx].t = tpoint;
                    Coef_[pidx].f    = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                    Coef_[pidx].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                    Coef_[pidx].fdsk = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                }
            }

            // Copy the coefficients to the buffer
            Coef.clear();
            int totalFeature = 0;
            for(int pidx = 0; pidx < pointsCount; pidx++)
            {
                LidarCoef &coef = Coef_[pidx];
                if (coef.t >= 0)
                {
                    Coef.push_back(coef);
                    Coef.back().ptIdx = totalFeature;
                    totalFeature++;

                }
            }
        }
    }

    void Deskew(GaussianProcessPtr &traj, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudDeskewedInB)
    {
        int Npoints = cloudRaw->size();
        cloudDeskewedInB->resize(Npoints);

        SE3d T_Be_W = traj->pose(cloudRaw->points.back().t).inverse();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int pidx = 0; pidx < Npoints; pidx++)
        {
            PointXYZIT &pi = cloudRaw->points[pidx];
            PointXYZI  &po = cloudDeskewedInB->points[pidx];

            double ts = pi.t;
            // if (!traj->TimeInInterval(tp))
            // {
            //     po.x = 0;
            //     po.y = 0;
            //     po.z = 0;
            //     // po.t = pi.t;
            //     po.intensity = pi.intensity;
            // }
            // else
            // {
                SE3d T_Be_Bs = T_Be_W*traj->pose(ts);

                Vector3d pinBs(pi.x, pi.y, pi.z);
                Vector3d pinBe = T_Be_Bs*pinBs;

                po.x = pinBe.x();
                po.y = pinBe.y();
                po.z = pinBe.z();
                // po.t = pi.t;
                po.intensity = pi.intensity;
            // }
        }
    }

    void CreateCeresProblem(ceres::Problem &problem, ceres::Solver::Options &options, ceres::Solver::Summary &summary,
                            GaussianProcessPtr &localTraj, double fixed_start, double fixed_end)
    {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 50;

        int KNOTS = localTraj->getNumKnots();

        // Add the parameter blocks for rotation
        for (int kidx = 0; kidx < KNOTS; kidx++)
        {
            problem.AddParameterBlock(localTraj->getKnotSO3(kidx).data(), 4, new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>());
            problem.AddParameterBlock(localTraj->getKnotOmg(kidx).data(), 3);
            problem.AddParameterBlock(localTraj->getKnotPos(kidx).data(), 3);
            problem.AddParameterBlock(localTraj->getKnotVel(kidx).data(), 3);
            problem.AddParameterBlock(localTraj->getKnotAcc(kidx).data(), 3);
        }

        // Fix the knots
        if (fixed_start >= 0)
            for (int kidx = 0; kidx < KNOTS; kidx++)
            {
                if (localTraj->getKnotTime(kidx) <= localTraj->getMinTime() + fixed_start)
                {
                    problem.SetParameterBlockConstant(localTraj->getKnotSO3(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotOmg(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotPos(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotVel(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotAcc(kidx).data());
                    // printf("Fixed knot %d\n", kidx);
                }
            }

        if (fixed_end >= 0)
            for (int kidx = 0; kidx < KNOTS; kidx++)
            {
                if (localTraj->getKnotTime(kidx) >= localTraj->getMaxTime() - fixed_end)
                {
                    problem.SetParameterBlockConstant(localTraj->getKnotSO3(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotOmg(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotPos(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotVel(kidx).data());
                    problem.SetParameterBlockConstant(localTraj->getKnotAcc(kidx).data());
                    // printf("Fixed knot %d\n", kidx);
                }
            }
    }

    void AddLidarFactors(vector<LidarCoef> &Coef, GaussianProcessPtr &traj, ceres::Problem &problem,
                         vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        static int skip = -1;
        for (auto &coef : Coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;

            skip++;
            if (skip % lidar_ds_rate != 0)
                continue;

            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;

            vector<double *> factor_param_blocks;

            // Add the parameter blocks for rotation
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            double ld_loss_thres = -1.0;
            nh_ptr->getParam("ld_loss_thres", ld_loss_thres);
            ceres::LossFunction *lidar_loss_function = ld_loss_thres <= 0 ? NULL : new ceres::HuberLoss(ld_loss_thres);
            ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.finW, coef.f, coef.n, lidar_weight*coef.plnrty, traj->getDt(), s);
            auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
            res_ids_lidar.push_back(res);

            // printf("Adding lidar factor %d. s: %f. u: %d, dt: %f\n", res_ids_lidar.size(), s, u, traj->getDt());
        }
    }

    void AddPosePriorFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                             vector<ceres::internal::ResidualBlock *> &res_ids_pose)
    {
        double dt_ = traj->getDt()/7;
        // Add the pose factors with priors sampled from previous spline
        for (double t = traj->getMinTime(); t < traj->getMaxTime(); t+=dt_)
        {
            // Continue if sample is in the window
            if (!traj->TimeInInterval(t, 1e-6))
                continue;

            auto   us = traj->computeTimeIndex(t);
            int    u  = us.first;
            double s  = us.second;

            vector<double *> factor_param_blocks;
            // Find the coupled poses
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            double pp_loss_thres = -1.0;
            nh_ptr->getParam("pp_loss_thres", pp_loss_thres);
            ceres::LossFunction *pose_loss_function = pp_loss_thres <= 0 ? NULL : new ceres::CauchyLoss(pp_loss_thres);
            ceres::CostFunction *cost_function = new GPPoseFactor(traj->pose(t), ppSigmaR, ppSigmaP, traj->getDt(), s);
            auto res_block = problem.AddResidualBlock(cost_function, pose_loss_function, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }
    }

    void AddMotionPriorFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                               vector<ceres::internal::ResidualBlock *> &res_ids_gp)
    {
        // Add GP factors between consecutive knots
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            // Create the factors
            double mp_loss_thres = -1;
            nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(mpSigmaR, mpSigmaP, traj->getDt());
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
            res_ids_gp.push_back(res_block);
        }

        // // Add the GP factors based on knot difference
        // for (int kidx = 0; kidx < traj->getNumKnots() - DK; kidx++)
        // {
        //     double ts = traj->getKnotTime(kidx) + tshift;
        //     double tf = traj->getKnotTime(kidx + DK) + tshift;

        //     if (!traj->TimeInInterval(ts, 1e-6) || !traj->TimeInInterval(tf, 1e-6))
        //         continue;

        //     // Find the coupled control points
        //     auto   uss = traj->computeTimeIndex(ts);
        //     int    us  = uss.first;
        //     double ss  = uss.second;

        //     auto   usf = traj->computeTimeIndex(tf);
        //     int    uf  = usf.first;
        //     double sf  = usf.second;

        //     // Confirm that basea and baseb are DK knots apart
        //     ROS_ASSERT(uf - us == DK && DK > 1);

        //     vector<double *> factor_param_blocks;

        //     // Add the parameter blocks
        //     for (int knot_idx = us; knot_idx < us + 2; knot_idx++)
        //     {
        //         factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        //     }

        //     for (int knot_idx = uf; knot_idx < uf + 2; knot_idx++)
        //     {
        //         factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
        //         factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
        //     }

        //     // Create the factor
        //     double mp_loss_thres = -1;
        //     nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
        //     ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
        //     ceres::CostFunction *cost_function = new GPMotionPriorFactor(mpSigmaR, mpSigmaP, traj->getDt(), ss, sf, tf - ts);
        //     auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);
        //     res_ids_gp.push_back(res_block);
        // }
    }

    void AddSmoothnessFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                              vector<ceres::internal::ResidualBlock *> &res_ids_sm)
    {
        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - 2; kidx++)
        {
            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = kidx; knot_idx < kidx + 3; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
            }

            // Create the factor
            double sm_loss_thres = -1;
            nh_ptr->getParam("sm_loss_thres", sm_loss_thres);
            ceres::LossFunction *sm_loss_function = sm_loss_thres <= 0 ? NULL : new ceres::HuberLoss(sm_loss_thres);
            ceres::CostFunction *cost_function = new GPSmoothnessFactor(smSigmaR, smSigmaP, traj->getDt());
            auto res_block = problem.AddResidualBlock(cost_function, sm_loss_function, factor_param_blocks);
            res_ids_sm.push_back(res_block);
        }
    }

    void AddExtrinsicPoseFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                                 vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        // Add the extrinsic factors at fix intervals
        for (double t = traj->getMinTime(); t < traj->getMaxTime() - 0.05; t += 0.05)
        {
            
        }
    }

    void AddAutodiffGPMPFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_gp;

        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - DK; kidx++)
        {
            double ts = traj->getKnotTime(kidx) + tshift;
            double tf = traj->getKnotTime(kidx + DK) + tshift;

            if (!traj->TimeInInterval(ts, 1e-6) || !traj->TimeInInterval(tf, 1e-6))
                continue;

            // Find the coupled control points
            auto   uss = traj->computeTimeIndex(ts);
            int    us  = uss.first;
            double ss  = uss.second;

            auto   usf = traj->computeTimeIndex(tf);
            int    uf  = usf.first;
            double sf  = usf.second;

            // Confirm that basea and baseb are DK knots apart
            ROS_ASSERT(uf - us == DK && DK > 1);

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            GPMotionPriorFactorAutodiff *GPMPFactor = new GPMotionPriorFactorAutodiff(mpSigmaR, mpSigmaP, traj->getDt(), ss, sf, tf - ts);
            auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPMotionPriorFactorAutodiff>(GPMPFactor);
            cost_function->SetNumResiduals(15);

            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = us; knot_idx < us + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());

                cost_function->AddParameterBlock(4);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
            }

            for (int knot_idx = uf; knot_idx < uf + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());

                cost_function->AddParameterBlock(4);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
            }

            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_gp.push_back(res_block);
        }

        gpmpFactorMeta.so3_parameter_blocks = so3_param;
        gpmpFactorMeta.r3_parameter_blocks = r3_param;
        gpmpFactorMeta.residual_blocks = res_ids_gp;

        // printf("Autodiff params: %d, %d, %d, %d, %d, %d\n",
        //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
        //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
        //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());

    }

    void AddAnalyticGPMPFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_gp;

        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - DK; kidx++)
        {
            double ts = traj->getKnotTime(kidx) + tshift;
            double tf = traj->getKnotTime(kidx + DK) + tshift;

            if (!traj->TimeInInterval(ts, 1e-6) || !traj->TimeInInterval(tf, 1e-6))
                continue;

            // Find the coupled control points
            auto   uss = traj->computeTimeIndex(ts);
            int    us  = uss.first;
            double ss  = uss.second;

            auto   usf = traj->computeTimeIndex(tf);
            int    uf  = usf.first;
            double sf  = usf.second;

            // Confirm that basea and baseb are DK knots apart
            ROS_ASSERT(uf - us == DK && DK > 1);

            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = us; knot_idx < us + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            for (int knot_idx = uf; knot_idx < uf + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorFactor(mpSigmaR, mpSigmaP, traj->getDt(), ss, sf, tf - ts);
            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_gp.push_back(res_block);
        }
        
        gpmpFactorMeta.so3_parameter_blocks = so3_param;
        gpmpFactorMeta.r3_parameter_blocks = r3_param;
        gpmpFactorMeta.residual_blocks = res_ids_gp;

        // printf("Analytic params: %d, %d, %d, %d, %d, %d\n",
        //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
        //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
        //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
    }

    void AddAutodiffGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_gp;

        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            GPMotionPriorTwoKnotsFactorAutodiff *GPMPFactor = new GPMotionPriorTwoKnotsFactorAutodiff(mpSigmaR, mpSigmaP, traj->getDt());
            auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPMotionPriorTwoKnotsFactorAutodiff>(GPMPFactor);
            cost_function->SetNumResiduals(15);

            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());

                cost_function->AddParameterBlock(4);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
            }

            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_gp.push_back(res_block);
        }

        gpmpFactorMeta.so3_parameter_blocks = so3_param;
        gpmpFactorMeta.r3_parameter_blocks = r3_param;
        gpmpFactorMeta.residual_blocks = res_ids_gp;

        // printf("Autodiff params: %d, %d, %d, %d, %d, %d\n",
        //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
        //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
        //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());

    }

    void AddAnalyticGPMP2KFactor(GaussianProcessPtr &traj, ceres::Problem &problem, FactorMeta &gpmpFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_gp;

        // Add GP factors between consecutive knots
        for (int kidx = 0; kidx < traj->getNumKnots() - 1; kidx++)
        {
            vector<double *> factor_param_blocks;
            // Add the parameter blocks
            for (int knot_idx = kidx; knot_idx < kidx + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            // Create the factors
            double mp_loss_thres = -1;
            nh_ptr->getParam("mp_loss_thres", mp_loss_thres);
            ceres::LossFunction *mp_loss_function = mp_loss_thres <= 0 ? NULL : new ceres::HuberLoss(mp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorTwoKnotsFactor(mpSigmaR, mpSigmaP, traj->getDt());
            auto res_block = problem.AddResidualBlock(cost_function, mp_loss_function, factor_param_blocks);

            res_ids_gp.push_back(res_block);
        }
        
        gpmpFactorMeta.so3_parameter_blocks = so3_param;
        gpmpFactorMeta.r3_parameter_blocks = r3_param;
        gpmpFactorMeta.residual_blocks = res_ids_gp;

        // printf("Analytic params: %d, %d, %d, %d, %d, %d\n",
        //         so3_param.size(), gpmpFactorMeta.so3_parameter_blocks.size(),
        //         r3_param.size(), gpmpFactorMeta.r3_parameter_blocks.size(),
        //         res_ids_gp.size(), gpmpFactorMeta.residual_blocks.size());
    }

    void AddAutodiffGPPoseFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                                 FactorMeta &gpposeFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_pose;

        double dt_ = traj->getDt()/7;
        // Add the pose factors with priors sampled from previous spline
        for (double t = traj->getMinTime(); t < traj->getMaxTime(); t+=dt_)
        {
            // Continue if sample is in the window
            if (!traj->TimeInInterval(t, 1e-6))
                continue;

            auto   us = traj->computeTimeIndex(t);
            int    u  = us.first;
            double s  = us.second;

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            GPPoseFactorAutodiff *GPPoseFactor = new GPPoseFactorAutodiff(traj->pose(t), 100.0, 100.0, traj->getDt(), s);
            auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPPoseFactorAutodiff>(GPPoseFactor);
            cost_function->SetNumResiduals(6);

            vector<double *> factor_param_blocks;
            // Find the coupled poses
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());

                cost_function->AddParameterBlock(4);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
            }

            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }

        gpposeFactorMeta.so3_parameter_blocks = so3_param;
        gpposeFactorMeta.r3_parameter_blocks = r3_param;
        gpposeFactorMeta.residual_blocks = res_ids_pose;
    }

    void AddAnalyticGPPoseFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                                 FactorMeta &gpposeFactorMeta)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_pose;

        double dt_ = traj->getDt()/7;
        // Add the pose factors with priors sampled from previous spline
        for (double t = traj->getMinTime(); t < traj->getMaxTime(); t+=dt_)
        {
            // Continue if sample is in the window
            if (!traj->TimeInInterval(t, 1e-6))
                continue;

            auto   us = traj->computeTimeIndex(t);
            int    u  = us.first;
            double s  = us.second;

            vector<double *> factor_param_blocks;
            // Find the coupled poses
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            ceres::CostFunction *cost_function = new GPPoseFactor(traj->pose(t), 100.0, 100.0, traj->getDt(), s);
            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }

        gpposeFactorMeta.so3_parameter_blocks = so3_param;
        gpposeFactorMeta.r3_parameter_blocks = r3_param;
        gpposeFactorMeta.residual_blocks = res_ids_pose;
    }

    void AddAutodiffGPLidarFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                                  FactorMeta &gplidarFactorMeta,
                                  vector<LidarCoef> &Coef)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_lidar;

        int skip = -1;
        for (auto &coef : Coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;

            skip++;
            if (skip % lidar_ds_rate != 0)
                continue;
            
            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;

            double lidar_loss_thres = -1.0;
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            GPPointToPlaneFactorAutodiff *GPLidarFactor = new GPPointToPlaneFactorAutodiff(coef.finW, coef.f, coef.n, coef.plnrty, traj->getDt(), s);
            auto *cost_function = new ceres::DynamicAutoDiffCostFunction<GPPointToPlaneFactorAutodiff>(GPLidarFactor);
            cost_function->SetNumResiduals(1);

            vector<double *> factor_param_blocks;
            // Add the parameter blocks for rotation
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());

                cost_function->AddParameterBlock(4);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
                cost_function->AddParameterBlock(3);
            }
            
            auto res = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_lidar.push_back(res);
        }

        gplidarFactorMeta.so3_parameter_blocks = so3_param;
        gplidarFactorMeta.r3_parameter_blocks = r3_param;
        gplidarFactorMeta.residual_blocks = res_ids_lidar;
    }

    void AddAnalyticGPLidarFactor(GaussianProcessPtr &traj, ceres::Problem &problem,
                                  FactorMeta &gplidarFactorMeta,
                                  vector<LidarCoef> &Coef)
    {
        vector<double *> so3_param;
        vector<double *> r3_param;
        vector<ceres::ResidualBlockId> res_ids_lidar;

        int skip = -1;
        for (auto &coef : Coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!traj->TimeInInterval(coef.t, 1e-6))
                continue;

            skip++;
            if (skip % lidar_ds_rate != 0)
                continue;
            
            auto   us = traj->computeTimeIndex(coef.t);
            int    u  = us.first;
            double s  = us.second;

            vector<double *> factor_param_blocks;

            // Add the parameter blocks for rotation
            for (int knot_idx = u; knot_idx < u + 2; knot_idx++)
            {
                so3_param.push_back(traj->getKnotSO3(knot_idx).data());
                r3_param.push_back(traj->getKnotOmg(knot_idx).data());
                r3_param.push_back(traj->getKnotPos(knot_idx).data());
                r3_param.push_back(traj->getKnotVel(knot_idx).data());
                r3_param.push_back(traj->getKnotAcc(knot_idx).data());

                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }
            
            double lidar_loss_thres = -1.0;
            ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
            ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.finW, coef.f, coef.n, coef.plnrty, traj->getDt(), s);
            auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
            res_ids_lidar.push_back(res);
        }

        gplidarFactorMeta.so3_parameter_blocks = so3_param;
        gplidarFactorMeta.r3_parameter_blocks = r3_param;
        gplidarFactorMeta.residual_blocks = res_ids_lidar;
    }

    void TestAnalyticJacobian(ceres::Problem &problem, GaussianProcessPtr &localTraj, vector<LidarCoef> &Coef, const int &cidx)
    {
        for(int kidx = 0; kidx < localTraj->getNumKnots(); kidx++)
            localTraj->setKnot(kidx, StateStamped<double>(0, SO3d(Util::YPR2Quat(Vector3d(0.57, 43, 91)*kidx)),
                                                             Vector3d(0, 1, 2)*sin(kidx+1),
                                                             Vector3d(4, 5, 6)*cos(kidx+1),
                                                             Vector3d(7, 8, 9)*sin(kidx+1),
                                                             Vector3d(10, 11, 12)*cos(kidx+1)));

        // Motion priors
        {
            double time_autodiff;
            VectorXd residual_autodiff_;
            MatrixXd Jacobian_autodiff_;
            {
                // Test the autodiff Jacobian
                FactorMeta gpmpFactorMetaAutodiff;
                AddAutodiffGPMPFactor(localTraj, problem, gpmpFactorMetaAutodiff);

                if (gpmpFactorMetaAutodiff.parameter_blocks() == 0)
                    return;

                TicToc tt_autodiff;
                double cost_autodiff;
                vector <double> residual_autodiff;
                MatrixXd J_autodiff;

                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpmpFactorMetaAutodiff, 0,
                                      cost_autodiff, residual_autodiff, J_autodiff);

                RemoveResidualBlock(problem, gpmpFactorMetaAutodiff);

                printf(KCYN "Motion Prior Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_autodiff.rows(), J_autodiff.cols(),
                       gpmpFactorMetaAutodiff.parameter_blocks(),
                       cost_autodiff, time_autodiff = tt_autodiff.Toc());

                residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
                Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
                // Jacobian_autodiff_ = J_autodiff.block(0, 6 + 15*0,  15, 9);
                // Jacobian_autodiff_.block(0, 6, 6, 6) = J_autodiff.block(0, 15, 6, 6);
                // Jacobian_autodiff_.block(6, 0, 6, 6) = J_autodiff.block(0, 30, 6, 6);
                // Jacobian_autodiff_.block(6, 6, 6, 6) = J_autodiff.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
            }

            double time_analytic;
            VectorXd residual_analytic_;
            MatrixXd Jacobian_analytic_;
            {
                // Test the analytic Jacobian
                FactorMeta gpmpFactorMetaAnalytic;
                AddAnalyticGPMPFactor(localTraj, problem, gpmpFactorMetaAnalytic);

                if (gpmpFactorMetaAnalytic.parameter_blocks() == 0)
                    return;

                TicToc tt_analytic;
                double cost_analytic;
                vector <double> residual_analytic;
                MatrixXd J_analytic;
                
                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpmpFactorMetaAnalytic, 1,
                                      cost_analytic, residual_analytic, J_analytic);

                RemoveResidualBlock(problem, gpmpFactorMetaAnalytic);

                printf(KMAG "Motion Prior Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_analytic.rows(), J_analytic.cols(),
                       gpmpFactorMetaAnalytic.parameter_blocks(),
                       cost_analytic, time_analytic = tt_analytic.Toc());

                residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
                Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
                // Jacobian_analytic_ = J_analytic.block(0, 6 + 15*0,  15, 9);
                // Jacobian_analytic_.block(0, 0, 6, 6) = J_analytic.block(0, 0,  6, 6);
                // Jacobian_analytic_.block(0, 6, 6, 6) = J_analytic.block(0, 15, 6, 6);
                // Jacobian_analytic_.block(6, 0, 6, 6) = J_analytic.block(0, 30, 6, 6);
                // Jacobian_analytic_.block(6, 6, 6, 6) = J_analytic.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_analytic_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
            }

            // Compare the two jacobians
            VectorXd resdiff = residual_autodiff_ - residual_analytic_;
            MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;

            // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
            // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;

            // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
            //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();

            printf(KGRN "CIDX: %d. MotionPrior Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n" RESET,
                   cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
        }

        // Motion priors
        {
            double time_autodiff;
            VectorXd residual_autodiff_;
            MatrixXd Jacobian_autodiff_;
            {
                // Test the autodiff Jacobian
                FactorMeta gpmp2kFactorMetaAutodiff;
                AddAutodiffGPMP2KFactor(localTraj, problem, gpmp2kFactorMetaAutodiff);

                if (gpmp2kFactorMetaAutodiff.parameter_blocks() == 0)
                    return;

                TicToc tt_autodiff;
                double cost_autodiff;
                vector <double> residual_autodiff;
                MatrixXd J_autodiff;

                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpmp2kFactorMetaAutodiff, 0,
                                      cost_autodiff, residual_autodiff, J_autodiff);

                RemoveResidualBlock(problem, gpmp2kFactorMetaAutodiff);

                printf(KCYN "Motion Prior 2Knot Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_autodiff.rows(), J_autodiff.cols(),
                       gpmp2kFactorMetaAutodiff.parameter_blocks(),
                       cost_autodiff, time_autodiff = tt_autodiff.Toc());

                residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
                Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
                // Jacobian_autodiff_ = J_autodiff.block(0, 6 + 15*0,  15, 9);
                // Jacobian_autodiff_ = J_autodiff.block(0, 0, 6, 15);
                // Jacobian_autodiff_.block(6, 0, 6, 6) = J_autodiff.block(0, 30, 6, 6);
                // Jacobian_autodiff_.block(6, 6, 6, 6) = J_autodiff.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
            }

            double time_analytic;
            VectorXd residual_analytic_;
            MatrixXd Jacobian_analytic_;
            {
                // Test the analytic Jacobian
                FactorMeta gpmp2kFactorMetaAnalytic;
                AddAnalyticGPMP2KFactor(localTraj, problem, gpmp2kFactorMetaAnalytic);

                if (gpmp2kFactorMetaAnalytic.parameter_blocks() == 0)
                    return;

                TicToc tt_analytic;
                double cost_analytic;
                vector <double> residual_analytic;
                MatrixXd J_analytic;
                
                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpmp2kFactorMetaAnalytic, 1,
                                      cost_analytic, residual_analytic, J_analytic);

                RemoveResidualBlock(problem, gpmp2kFactorMetaAnalytic);

                printf(KMAG "Motion Prior 2Knot Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_analytic.rows(), J_analytic.cols(),
                       gpmp2kFactorMetaAnalytic.parameter_blocks(),
                       cost_analytic, time_analytic = tt_analytic.Toc());

                residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
                Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
                // Jacobian_analytic_ = J_analytic.block(0, 6 + 15*0,  15, 9);
                // Jacobian_analytic_ = J_analytic.block(0, 0,  6, 15);
                // Jacobian_analytic_.block(0, 6, 6, 6) = J_analytic.block(0, 15, 6, 6);
                // Jacobian_analytic_.block(6, 0, 6, 6) = J_analytic.block(0, 30, 6, 6);
                // Jacobian_analytic_.block(6, 6, 6, 6) = J_analytic.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_analytic_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
            }

            // Compare the two jacobians
            VectorXd resdiff = residual_autodiff_ - residual_analytic_;
            MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;

            // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
            // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;

            // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
            //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();

            printf(KGRN "CIDX: %d. MotionPrior 2K Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n" RESET,
                   cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
        }

        // Pose factors
        {
            double time_autodiff;
            VectorXd residual_autodiff_;
            MatrixXd Jacobian_autodiff_;
            {
                // Test the autodiff Jacobian
                FactorMeta gpposeFactorMetaAutodiff;
                AddAutodiffGPPoseFactor(localTraj, problem, gpposeFactorMetaAutodiff);

                if (gpposeFactorMetaAutodiff.parameter_blocks() == 0)
                    return;

                TicToc tt_autodiff;
                double cost_autodiff;
                vector <double> residual_autodiff;
                MatrixXd J_autodiff;

                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpposeFactorMetaAutodiff, 0,
                                      cost_autodiff, residual_autodiff, J_autodiff);

                RemoveResidualBlock(problem, gpposeFactorMetaAutodiff);

                printf(KCYN "Pose Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_autodiff.rows(), J_autodiff.cols(),
                       gpposeFactorMetaAutodiff.parameter_blocks(),
                       cost_autodiff, time_autodiff = tt_autodiff.Toc());

                residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
                Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
                // Jacobian_autodiff_ = J_autodiff.block(0, 6 + 15*0,  15, 9);
                // Jacobian_autodiff_.block(0, 6, 6, 6) = J_autodiff.block(0, 15, 6, 6);
                // Jacobian_autodiff_.block(6, 0, 6, 6) = J_autodiff.block(0, 30, 6, 6);
                // Jacobian_autodiff_.block(6, 6, 6, 6) = J_autodiff.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
            }

            double time_analytic;
            VectorXd residual_analytic_;
            MatrixXd Jacobian_analytic_;
            {
                // Test the analytic Jacobian
                FactorMeta gpposeFactorMetaAnalytic;
                AddAnalyticGPPoseFactor(localTraj, problem, gpposeFactorMetaAnalytic);

                if (gpposeFactorMetaAnalytic.parameter_blocks() == 0)
                    return;

                TicToc tt_analytic;
                double cost_analytic;
                vector <double> residual_analytic;
                MatrixXd J_analytic;
                
                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gpposeFactorMetaAnalytic, 1,
                                      cost_analytic, residual_analytic, J_analytic);

                RemoveResidualBlock(problem, gpposeFactorMetaAnalytic);

                printf(KMAG "Pose Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_analytic.rows(), J_analytic.cols(),
                       gpposeFactorMetaAnalytic.parameter_blocks(),
                       cost_analytic, time_analytic = tt_analytic.Toc());

                residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
                Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
                // Jacobian_analytic_ = J_analytic.block(0, 6 + 15*0,  15, 9);
                // Jacobian_analytic_.block(0, 0, 6, 6) = J_analytic.block(0, 0,  6, 6);
                // Jacobian_analytic_.block(0, 6, 6, 6) = J_analytic.block(0, 15, 6, 6);
                // Jacobian_analytic_.block(6, 0, 6, 6) = J_analytic.block(0, 30, 6, 6);
                // Jacobian_analytic_.block(6, 6, 6, 6) = J_analytic.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_analytic_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
            }

            // Compare the two jacobians
            VectorXd resdiff = residual_autodiff_ - residual_analytic_;
            MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;

            // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
            // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;

            // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
            //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();

            printf(KGRN "CIDX: %d. Pose Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n" RESET,
                   cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
        }

        // Lidar factors
        {
            double time_autodiff;
            VectorXd residual_autodiff_;
            MatrixXd Jacobian_autodiff_;
            {
                // Test the autodiff Jacobian
                FactorMeta gplidarFactorMetaAutodiff;
                AddAutodiffGPLidarFactor(localTraj, problem, gplidarFactorMetaAutodiff, Coef);

                if (gplidarFactorMetaAutodiff.parameter_blocks() == 0)
                    return;

                TicToc tt_autodiff;
                double cost_autodiff;
                vector <double> residual_autodiff;
                MatrixXd J_autodiff;

                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gplidarFactorMetaAutodiff, 0,
                                      cost_autodiff, residual_autodiff, J_autodiff);

                RemoveResidualBlock(problem, gplidarFactorMetaAutodiff);

                printf(KCYN "Lidar Autodiff Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_autodiff.rows(), J_autodiff.cols(),
                       gplidarFactorMetaAutodiff.parameter_blocks(),
                       cost_autodiff, time_autodiff = tt_autodiff.Toc());

                residual_autodiff_ = Eigen::Map<Eigen::VectorXd>(residual_autodiff.data(), residual_autodiff.size());
                Jacobian_autodiff_ = J_autodiff;//MatrixXd(15, 9).setZero();
                // Jacobian_autodiff_ = J_autodiff.block(0, 6 + 15*0,  15, 9);
                // Jacobian_autodiff_.block(0, 6, 6, 6) = J_autodiff.block(0, 15, 6, 6);
                // Jacobian_autodiff_.block(6, 0, 6, 6) = J_autodiff.block(0, 30, 6, 6);
                // Jacobian_autodiff_.block(6, 6, 6, 6) = J_autodiff.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_autodiff_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_autodiff_ << RESET << endl;
            }

            double time_analytic;
            VectorXd residual_analytic_;
            MatrixXd Jacobian_analytic_;
            {
                // Test the analytic Jacobian
                FactorMeta gplidarFactorMetaAnalytic;
                AddAnalyticGPLidarFactor(localTraj, problem, gplidarFactorMetaAnalytic, Coef);

                if (gplidarFactorMetaAnalytic.parameter_blocks() == 0)
                    return;

                TicToc tt_analytic;
                double cost_analytic;
                vector <double> residual_analytic;
                MatrixXd J_analytic;
                
                int count = 100;
                while(count-- > 0)
                    GetFactorJacobian(problem, gplidarFactorMetaAnalytic, 1,
                                      cost_analytic, residual_analytic, J_analytic);

                RemoveResidualBlock(problem, gplidarFactorMetaAnalytic);

                printf(KMAG "Lidar Analytic Jacobian: Size %2d %2d. Params: %d. Cost: %f. Time: %f.\n",
                       J_analytic.rows(), J_analytic.cols(),
                       gplidarFactorMetaAnalytic.parameter_blocks(),
                       cost_analytic, time_analytic = tt_analytic.Toc());

                residual_analytic_ = Eigen::Map<Eigen::VectorXd>(residual_analytic.data(), residual_analytic.size());
                Jacobian_analytic_ = J_analytic;//MatrixXd(15, 9).setZero();
                // Jacobian_analytic_ = J_analytic.block(0, 6 + 15*0,  15, 9);
                // Jacobian_analytic_.block(0, 0, 6, 6) = J_analytic.block(0, 0,  6, 6);
                // Jacobian_analytic_.block(0, 6, 6, 6) = J_analytic.block(0, 15, 6, 6);
                // Jacobian_analytic_.block(6, 0, 6, 6) = J_analytic.block(0, 30, 6, 6);
                // Jacobian_analytic_.block(6, 6, 6, 6) = J_analytic.block(0, 45, 6, 6);

                // cout << "residual:\n" << residual_analytic_.transpose() << endl;
                // cout << "jacobian:\n" << Jacobian_analytic_ << RESET << endl;
            }

            // Compare the two jacobians
            VectorXd resdiff = residual_autodiff_ - residual_analytic_;
            MatrixXd jcbdiff = Jacobian_autodiff_ - Jacobian_analytic_;

            // cout << KRED "residual diff:\n" RESET << resdiff.transpose() << endl;
            // cout << KRED "jacobian diff:\n" RESET << jcbdiff << endl;

            // if (maxCoef < jcbdiff.cwiseAbs().maxCoeff() && cidx != 0)
            //     maxCoef = jcbdiff.cwiseAbs().maxCoeff();

            printf(KGRN "CIDX: %d. Lidar Jacobian max error: %.4f. Time: %.3f, %.3f. Ratio: %.0f\%\n\n" RESET,
                   cidx, jcbdiff.cwiseAbs().maxCoeff(), time_autodiff, time_analytic, time_autodiff/time_analytic*100);
        }
    }
    
    void ChopTheClouds(const vector<CloudXYZITPtr> &clouds)
    {
        int Ncloud = clouds.size();
        static int lastCloudIdx;
        static int lastPointIdx;
        static double lastCutTime = clouds.front()->points.front().t;

        while(ros::ok())
        {
            if (cloud_seg_buf.size() < 57)
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
                        //        cidx, cloudSeg->size(), tp, lastCutTime, lastCutTime + deltaT);
                        if (tp < lastCutTime)
                        {
                            if(pidx == clouds[cidx]->size() - 1)
                                lastPointIdx = -1;
                            
                            continue;
                        }
                        
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
                    std::lock_guard<mutex> lg(cloud_seg_buf_mtx);
                    cloud_seg_buf.push_back(cloudSeg);
                    
                    // printf("Chopper cloud_seg_buf: %d, %d. cloudSeg: %d. Intv: %f, %f. CutTime: %f, %f\n",
                    //         cloud_seg_buf.size(),
                    //         cloud_seg_buf.back()->size(),
                    //         cloudSeg->size(),
                    //         cloudSeg->points.front().t,
                    //         cloudSeg->points.back().t,
                    //         lastCutTime, lastCutTime + deltaT);
                            
                    lastCutTime += deltaT;    
                }
                else
                {
                    printf("No more points.\n");
                    this_thread::sleep_for(chrono::milliseconds(1000));
                }
            }
        }
    }

    void FindTraj(const KdFLANNPtr &kdTreeMap, const CloudXYZIPtr priormap,
                  const vector<CloudXYZITPtr> &clouds)
    {
        int Ncloud = clouds.size();
        // CloudPosePtr posePrior = CloudPosePtr(new CloudPose());
        // posePrior->resize(Ncloud);

        GaussianProcessPtr traj(new GaussianProcess(deltaT));
        traj->setStartTime(clouds.front()->points.front().t);
        traj->setKnot(0, StateStamped(clouds.front()->points.front().t, T_W_Li0));

        static deque<CloudXYZITPtr> swCloudSeg;
        static deque<CloudXYZIPtr > swCloudSegUndi;
        static deque<CloudXYZIPtr > swCloudSegUndiInW;
        deque<vector<LidarCoef>> swCloudCoef;

        // Check the buffer
        while(ros::ok())
        {
            // Step 0: Poll and Extract the cloud segment -------------------------------------------------------------

            CloudXYZITPtr cloudSeg;
            if(cloud_seg_buf.size() != 0)
            {
                std::lock_guard<mutex> lg(cloud_seg_buf_mtx);
                cloudSeg = cloud_seg_buf.front();
                cloud_seg_buf.pop_front();
            }
            else
            {
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }

            TicToc tt_loop;
            TicToc tt_preopt;



            // Step 1: Extend the trajectory to the new end time, trim the time by millisecond ------------------------
            
            double TSWEND = cloudSeg->points.back().t;
            // Extend the knot by propagation
            while(traj->getMaxTime() < TSWEND)
                traj->extendOneKnot();

            // Downsample the input pointcloud
            pcl::UniformSampling<PointXYZIT> downsampler;
            downsampler.setRadiusSearch(ds_size);
            downsampler.setInputCloud(cloudSeg);
            downsampler.filter(*cloudSeg);



            // Step 2: Admit the cloud segment ------------------------------------------------------------------------

            swCloudSeg.push_back(cloudSeg);
            swCloudSegUndi.push_back(CloudXYZIPtr(new CloudXYZI()));
            swCloudSegUndiInW.push_back(CloudXYZIPtr(new CloudXYZI()));
            swCloudCoef.push_back(vector<LidarCoef>());

            // Shift the sliding window if length exceeds threshold
            if (swCloudSeg.size() > WINDOW_SIZE)
            {
                swCloudSeg.pop_front();
                swCloudSegUndi.pop_front();
                swCloudSegUndiInW.pop_front();
                swCloudCoef.pop_front();
            }

            // No need to deskew for the first interval
            pcl::copyPointCloud(*swCloudSeg.back(), *swCloudSegUndi.back());

            // Transform cloud to the world frame for association
            SE3d pose = traj->pose(TSWEND);
            pcl::transformPointCloud(*swCloudSegUndi.back(), *swCloudSegUndiInW.back(), pose.translation(), pose.so3().unit_quaternion());

            // Step 2.1: Associate the last pointcloud with the map
            Associate(kdTreeMap, priormap, swCloudSeg.back(), swCloudSegUndi.back(), swCloudSegUndiInW.back(), swCloudCoef.back());          

            // Create a local trajectory for optimization
            GaussianProcessPtr localTraj(new GaussianProcess(deltaT));
            int    umin = traj->computeTimeIndex(max(traj->getMinTime(), swCloudSeg.front()->points.front().t)).first;
            double tmin = traj->getKnotTime(umin);
            double tmax = min(traj->getMaxTime(), TSWEND);
            // Copy the knots {umin, umin+1, ...} from traj to localtraj
            for(int kidx = umin; kidx < traj->getNumKnots(); kidx++)
                localTraj->extendOneKnot(traj->getKnot(kidx));
            // Reset the start time
            localTraj->setStartTime(tmin);
            // printf("umin: %d. tmin: %f. tmax: %f. LcKnots: %d. TrajKnots: %d\n",
            //         umin, tmin, tmax, localTraj->getNumKnots(), traj->getNumKnots());
            StateStamped Xt0 = traj->getStateAt(TSWEND);
            // if (Xt0.P.norm() > 100.0)
            // {   
            //     for(int kidx = 0; kidx < localTraj->getNumKnots(); kidx++)
            //     {
            //         printf("Knot: %d\n", kidx);
            //         Quaternd q = Xt0.R.unit_quaternion();
            //         cout << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << endl;
            //         cout << Xt0.O.transpose() << endl;
            //         cout << Xt0.P.transpose() << endl;
            //         cout << Xt0.V.transpose() << endl;
            //         cout << Xt0.A.transpose() << endl;
            //     }
            //     exit(-1);
            // }

            int WDZ = min(int(swCloudSeg.size()), WINDOW_SIZE);
            // Loop if the sliding window is not yet long enough
            // if (WDZ < WINDOW_SIZE)
            //     continue;

            tt_preopt.Toc();



            // Step 3: iterative optimization -------------------------------------------------------------------------

            vector<string> report(max_gniter);
            
            static int optnum = -1; optnum++;
            int outeritr = 0;
            while(outeritr < max_gniter && traj->getMaxTime() > SKIPPED_TIME)
            {
                TicToc tt_build;

                // Step 3.3: Create the ceres problem and add the knots to the param list

                // Create the ceres problem
                ceres::Problem problem;
                ceres::Solver::Options options;
                ceres::Solver::Summary summary;
                if(use_ceres)
                    CreateCeresProblem(problem, options, summary, localTraj, fixed_start, fixed_end);

                // Test if the Jacobian works
                // TestAnalyticJacobian(problem, localTraj, swCloudCoef[0], traj->getNumKnots());
                // continue;

                // Step 3.4: Add the lidar factors
                double cost_lidar_begin = -1;
                double cost_lidar_final = -1;
                vector<ceres::internal::ResidualBlock *> res_ids_lidar;
                if (lidar_weight >= 0.0 && use_ceres)
                    for(int widx = 0; widx < WDZ; widx++)
                        AddLidarFactors(swCloudCoef[widx], localTraj, problem, res_ids_lidar);
                // else
                //     printf(KYEL "Skipping lidar factors.\n" RESET);

                // Step 3.5: Add pose prior factors
                double cost_pose_begin = -1;
                double cost_pose_final = -1;
                vector<ceres::internal::ResidualBlock *> res_ids_pose;
                // if (ppSigmaR >= 0.0 && ppSigmaP >= 0.0 && use_ceres)
                //     AddPosePriorFactors(localTraj, problem, res_ids_pose);
                // else
                //     printf(KYEL "Skipping pose priors.\n" RESET);

                // Step 3.6: Add motion prior factors
                double cost_mp_begin = -1;
                double cost_mp_final = -1;
                vector<ceres::internal::ResidualBlock *> res_ids_mp;
                if(mpSigmaR >= 0.0 && mpSigmaP >= 0.0 && use_ceres)
                    AddMotionPriorFactors(localTraj, problem, res_ids_mp);
                // else
                //     printf(KYEL "Skipping motion prior factors.\n" RESET);

                // Step 3.7: Add smoothness constraints factors
                double cost_sm_begin = -1;
                double cost_sm_final = -1;
                vector<ceres::internal::ResidualBlock *> res_ids_sm;
                // if(smSigmaR >= 0.0 && smSigmaP >= 0.0 && use_ceres)
                //     AddSmoothnessFactors(localTraj, problem, res_ids_sm);
                // else
                //     printf(KYEL "Skipping smoothness factors.\n" RESET);
                
                tt_build.Toc();


                TicToc tt_solve;
                
                if(use_ceres)
                {
                    // Initial cost
                    Util::ComputeCeresCost(res_ids_lidar, cost_lidar_begin, problem);
                    Util::ComputeCeresCost(res_ids_pose, cost_pose_begin, problem);
                    Util::ComputeCeresCost(res_ids_mp, cost_mp_begin, problem);
                    Util::ComputeCeresCost(res_ids_sm, cost_sm_begin, problem);

                    // Solve and visualize:
                    ceres::Solve(options, &problem, &summary);

                    // Final cost
                    Util::ComputeCeresCost(res_ids_lidar, cost_lidar_final, problem);
                    Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);
                    Util::ComputeCeresCost(res_ids_mp, cost_mp_final, problem);
                    Util::ComputeCeresCost(res_ids_sm, cost_sm_final, problem);
                }
                else
                    mySolver->Solve(localTraj, swCloudCoef, outeritr);

                tt_solve.Toc();
                
                TicToc tt_aftop;
                // Step X: Copy the knots back to the global trajectory
                {
                    for(int kidx = 0; kidx < localTraj->getNumKnots(); kidx++)
                    {
                        double tgb = traj->getKnotTime(kidx + umin);
                        double tlc = localTraj->getKnotTime(kidx);
                        double ter = fabs(tlc - tgb);
                        ROS_ASSERT_MSG(ter < 1e-3, "Knot Time: %f, %f. Diff: %f.\n", tlc, tgb, ter);
                        traj->setKnot(kidx + umin, localTraj->getKnot(kidx));
                    }
                }

                // Deskew the point cloud
                for(int widx = 0; widx < WDZ; widx++)
                {
                    Deskew(traj, swCloudSeg[widx], swCloudSegUndi[widx]);

                    // Transform pointcloud to the world frame
                    double tend = swCloudSeg[widx]->points.back().t;
                    myTf tf_W_Be(traj->pose(tend));
                    pcl::transformPointCloud(*swCloudSegUndi[widx],
                                             *swCloudSegUndiInW[widx],
                                              tf_W_Be.pos, tf_W_Be.rot);
                    
                    // Associate between feature and map
                    Associate(kdTreeMap, priormap, swCloudSeg[widx], swCloudSegUndi[widx], swCloudSegUndiInW[widx], swCloudCoef[widx]);
                }

                StateStamped XtK = traj->getStateAt(TSWEND);

                // Sample and publish the slinding window trajectory
                CloudPosePtr poseSampled = CloudPosePtr(new CloudPose());
                for(double ts = localTraj->getMinTime(); ts < localTraj->getMaxTime(); ts += localTraj->getDt()/5)
                    if(localTraj->TimeInInterval(ts))
                        poseSampled->points.push_back(myTf(localTraj->pose(ts)).Pose6D(ts));
                static ros::Publisher swTrajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
                Util::publishCloud(swTrajPub, *poseSampled, ros::Time::now(), "world");

                CloudXYZIPtr assoc_cloud(new CloudXYZI());
                for (int widx = 0; widx < WDZ; widx++)
                    for(auto &coef : swCloudCoef[widx])
                        {
                            PointXYZI p;
                            p.x = coef.finW.x();
                            p.y = coef.finW.y();
                            p.z = coef.finW.z();
                            p.intensity = widx;
                            assoc_cloud->push_back(p);
                        }
                static ros::Publisher assocCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
                Util::publishCloud(assocCloudPub, *assoc_cloud, ros::Time::now(), "world");

                outeritr++;

                tt_aftop.Toc();

                // Print a report
                double swTs = swCloudSeg.front()->points.front().t;
                double swTe = swCloudSeg.back()->points.back().t;
                // double gpTs = localTraj->getMinTime();
                // double gpTe = localTraj->getMaxTime();
                report[outeritr-1] = 
                myprintf("%sGPMAPLO#%d. OItr: %2d / %2d. GNItr: %2d. Umin: %4d. TKnot: %6.3f -> %6.3f. TCloud: %6.3f -> %6.3f.\n"
                         "Tprop: %.0f. Tbuild: %.0f. Tslv: %.0f. Taftop: %.0f. Tlp: %.0f.\n"
                         "Factors: Lidar: %4d. Pose: %4d. Motion prior: %4d. Smoothness: %4d. Knots: %d / %d.\n"
                         "J0: %12.3f. Ldr: %9.3f. Pose: %9.3f. MP: %9.3f. SM: %9.3f\n"
                         "JK: %12.3f. Ldr: %9.3f. Pose: %9.3f. MP: %9.3f. SM: %9.3f\n"
                         "Pos0: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f\n"
                         "PosK: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f\n"
                         RESET,
                         outeritr == max_gniter ? KGRN : "", optnum,
                         outeritr, max_gniter, (int)(summary.iterations.size()), umin, tmin, tmax, swTs, swTe,
                         tt_preopt.GetLastStop(), tt_build.GetLastStop(), tt_solve.GetLastStop(), tt_aftop.GetLastStop(),
                         outeritr == max_gniter ? tt_loop.Toc() : -1.0,
                         res_ids_lidar.size(), res_ids_pose.size(), res_ids_mp.size(), res_ids_sm.size(), localTraj->getNumKnots(), traj->getNumKnots(),
                         summary.initial_cost, cost_lidar_begin, cost_pose_begin, cost_mp_begin, cost_sm_begin,
                         summary.final_cost, cost_lidar_final, cost_pose_final, cost_mp_final, cost_sm_final,
                         Xt0.P.x(), Xt0.P.y(), Xt0.P.z(), Xt0.V.x(), Xt0.V.y(), Xt0.V.z(),
                         XtK.P.x(), XtK.P.y(), XtK.P.z(), XtK.V.x(), XtK.V.y(), XtK.V.z());

                // Print the report
                if (outeritr == max_gniter)
                {
                    for(string &rep : report)
                        cout << rep;
                    cout << endl;
                }
            }
            // this_thread::sleep_for(chrono::milliseconds(100));
        }

        //     // Sleep for some time
        //     // this_thread::sleep_for(std::chrono::milliseconds(500));
        //     // std::cin.get();

    }
};
