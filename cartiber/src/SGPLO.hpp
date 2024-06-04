#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utility.h"

#include "factor/PoseAnalyticFactor.h"
#include "factor/GaussianPriorFactor.h"
#include "factor/GPPoseFactor.h"
#include "factor/GPMotionPriorFactor.h"
#include "factor/GPPointToPlaneFactor.h"

#include "GaussianProcess.hpp"

using NodeHandlePtr = boost::shared_ptr<ros::NodeHandle>;
// using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
typedef std::shared_ptr<GPLO> GPLOPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

class SGPLO
{
private:

    NodeHandlePtr nh_ptr;

    // How many point clouds to import into the sliding window
    int WINDOW_SIZE = 10;

    // Extrinsics of the lidars
    vector<SE3d> T_L0_Li;

    // Spline to model the trajectory of each lidar
    vector<GaussianProcessPtr> traj;

    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;
    double minKnnNbrDis = 0.1;

    // Rate of skipping the factors
    int lidar_ds_rate = 1;

    int SPLINE_N = 4;
    double deltaT = 0.1;

    int DK = 1;
    double tshift = 0.05;

    double wR = 10;
    double wP = 10;

    // Time to fix the knot
    double fixed_start = 0.0;
    double fixed_end = 0.0;

public:

    // Destructor
   ~SGPLO() {};

    SGPLO(NodeHandlePtr &nh_ptr_, vector<GaussianProcessPtr> &traj_, vector<SE3d> &T_L0_Li_)
        : nh_ptr(nh_ptr_), traj(traj_), T_L0_Li(T_L0_Li_)
    {
        // Trajectory estimate
        nh_ptr->getParam("SPLINE_N", SPLINE_N);
        nh_ptr->getParam("deltaT", deltaT);

        // Window size
        nh_ptr->getParam("WINDOW_SIZE", WINDOW_SIZE);

        // Trajectory estimate
        nh_ptr->getParam("fixed_start", fixed_start);
        nh_ptr->getParam("fixed_end", fixed_end);

        nh_ptr->getParam("DK", DK);
        nh_ptr->getParam("tshift", tshift);

        nh_ptr->getParam("wR", wR);
        nh_ptr->getParam("wP", wP);

        printf("Window size: %d. Fixes: <%f, >%f. GPinvt: %f, %d. Weight: %f, %f.\n",
                WINDOW_SIZE, fixed_start, fixed_end, tshift, DK, wR, wP);
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
                    totalFeature++;
                    Coef.push_back(coef);
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
            SE3d T_Be_Bs = T_Be_W*traj->pose(ts);

            Vector3d pinBs(pi.x, pi.y, pi.z);
            Vector3d pinBe = T_Be_Bs*pinBs;

            po.x = pinBe.x();
            po.y = pinBe.y();
            po.z = pinBe.z();
            // po.t = pi.t;
            po.intensity = pi.intensity;            
        }
    }

    void CreateCeresProblem(ceres::Problem &problem, ceres::Solver::Options &options, ceres::Solver::Summary &summary,
                            vector<GaussianProcessPtr> &localTraj, double fixed_start, double fixed_end)
    {
        int Nlidar = traj.size();
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 50;

        ceres::LocalParameterization *local_parameterization = new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>();
        for (int lidx = 0; lidx < Nlidar; lidx++)
        {
            int KNOTS = localTraj[lidx]->getNumKnots();
            
            // Add the parameter blocks for rotation
            for (int kidx = 0; kidx < KNOTS; kidx++)
            {
                problem.AddParameterBlock(localTraj[lidx]->getKnotSO3(kidx).data(), 4, local_parameterization);
                problem.AddParameterBlock(localTraj[lidx]->getKnotOmg(kidx).data(), 3);
                problem.AddParameterBlock(localTraj[lidx]->getKnotPos(kidx).data(), 3);
                problem.AddParameterBlock(localTraj[lidx]->getKnotVel(kidx).data(), 3);
                problem.AddParameterBlock(localTraj[lidx]->getKnotAcc(kidx).data(), 3);                
            }

            // Fix the knots
            if (fixed_start >= 0)
                for (int kidx = 0; kidx < KNOTS; kidx++)
                {
                    if (localTraj[lidx]->getKnotTime(kidx) <= localTraj[lidx]->getMinTime() + fixed_start)
                    {
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(kidx).data());
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(kidx).data());
                    }
                }

            if (fixed_end >= 0)
            {
                for (int kidx = 0; kidx < KNOTS; kidx++)
                {
                    if (localTraj[lidx]->getKnotTime(kidx) >= localTraj[lidx]->getMaxTime() - fixed_end)
                    {
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(KNOTS - 1 - kidx).data());
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(KNOTS - 1 - kidx).data());
                    }
                }
            }    
        }
    }

    void AddLidarFactors(vector<LidarCoef> &coef, GaussianProcessPtr &traj, ceres::Problem &problem,
                         vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        static int skip = -1;
        for (auto &coef : coef)
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
            
            double lidar_loss_thres = -1.0;
            ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
            ceres::CostFunction *cost_function = new GPPointToPlaneFactor(coef.finW, coef.f, coef.n, coef.plnrty, traj->getDt(), s);
            auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
            res_ids_lidar.push_back(res);
        }
    }

    void AddPosePriorFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                             vector<ceres::internal::ResidualBlock *> &res_ids_pose)
    {
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

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

            ceres::CostFunction *cost_function = new GPPoseFactor(traj->pose(t), 100.0, 100.0, traj->getDt(), s);
            auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }
    }

    void AddMotionPriorFactors(GaussianProcessPtr &traj, ceres::Problem &problem,
                               vector<ceres::internal::ResidualBlock *> &res_ids_gp)
    {
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
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            for (int knot_idx = uf; knot_idx < uf + 2; knot_idx++)
            {
                factor_param_blocks.push_back(traj->getKnotSO3(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotOmg(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotPos(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotVel(knot_idx).data());
                factor_param_blocks.push_back(traj->getKnotAcc(knot_idx).data());
            }

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(gp_loss_thres);
            ceres::CostFunction *cost_function = new GPMotionPriorFactor(wR, wP, traj->getDt(), ss, sf, tf - ts);

            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_gp.push_back(res_block);

            return;
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

    void FindTraj(const KdFLANNPtr &kdTreeMap, const CloudXYZIPtr priormap,
                  const vector<vector<CloudXYZITPtr>> &clouds)
    {
        int Nlidar = clouds.size();
        int Ncloud = clouds.front().size();
        CloudPosePtr posePrior = CloudPosePtr(new CloudPose());
        posePrior->resize(Ncloud);

        for(int cidx = 0; cidx < Ncloud && ros::ok(); cidx += WINDOW_SIZE/2)
        {
            // Index of the sliding window
            int gbStart = cidx;
            int gbFinal = min(cidx + WINDOW_SIZE, Ncloud);
            int WDZ = gbFinal - gbStart;

            // Get the pointclouds belonging to the primary cloud
            vector<vector<CloudXYZITPtr>> swClouds(Nlidar);
            vector<vector<CloudXYZIPtr>>  swCloudsDeskewed(Nlidar);
            vector<vector<CloudXYZIPtr>>  swCloudsDeskewedInW(Nlidar);

            // Extract the pointclouds on the sliding window
            for (int lidx = 0; lidx < Nlidar; lidx++)
                for (int gbIdx = gbStart; gbIdx < gbFinal; gbIdx++)
                {
                    swClouds[lidx].push_back(clouds[lidx][gbIdx]);
                    swCloudsDeskewed[lidx].push_back(CloudXYZIPtr(new CloudXYZI()));
                    swCloudsDeskewedInW[lidx].push_back(CloudXYZIPtr(new CloudXYZI()));
                }

            // Step 0: Deskew
            for (int lidx = 0; lidx < Nlidar; lidx++)
                for (int swIdx = 0; swIdx < WDZ; swIdx++)
                {
                    double te = swClouds[lidx][swIdx]->back().t;

                    // Deskew to the end time of the scan
                    // swCloudsDeskewed[lidx][swIdx] = CloudXYZIPtr(new CloudXYZI());
                    // swCloudsDeskewedInW[lidx][swIdx] = CloudXYZIPtr(new CloudXYZI());
                    Deskew(traj[lidx], swClouds[lidx][swIdx], swCloudsDeskewed[lidx][swIdx]);

                    // Transform pointcloud to the world frame
                    myTf tf_W_Be(traj[lidx]->pose(te));
                    pcl::transformPointCloud(*swCloudsDeskewed[lidx][swIdx],
                                             *swCloudsDeskewedInW[lidx][swIdx],
                                              tf_W_Be.pos, tf_W_Be.rot);
                }

            // Step 1: Associate
            vector<vector<vector<LidarCoef>>> Coef(Nlidar, vector<vector<LidarCoef>>(WDZ));
            for (int lidx = 0; lidx < Nlidar; lidx++)
                for (int swIdx = 0; swIdx < WDZ; swIdx++)
                    Associate(kdTreeMap, priormap,
                              swClouds[lidx][swIdx],
                              swCloudsDeskewed[lidx][swIdx],
                              swCloudsDeskewedInW[lidx][swIdx],
                              Coef[lidx][swIdx]);

            // Step 2: Build the optimization problem

            // Step 2.1: Copy the knots to the local trajectories
            vector<GaussianProcessPtr> localTraj(Nlidar);
            for (int lidx = 0; lidx < Nlidar; lidx++)
            {
                int    umin = traj[lidx]->computeTimeIndex(max(traj[lidx]->getMinTime(), swClouds[lidx].front()->points.front().t)).first;                
                double tmin = traj[lidx]->getKnotTime(umin);
                double tmax = min(traj[lidx]->getMaxTime(), swClouds[lidx].back()->points.back().t);

                // Find the knots related to this trajectory
                localTraj[lidx] = GaussianProcessPtr(new GaussianProcess(deltaT));
                localTraj[lidx]->setStartTime(tmin);
                localTraj[lidx]->extendKnotsTo(tmax);

                // Find the starting knot in traj and copy to localtraj
                for(int kidx = 0; kidx < localTraj[lidx]->getNumKnots(); kidx++)
                    localTraj[lidx]->setKnot(kidx, traj[lidx]->getKnot(kidx + umin));
            }

            // Step 2,2: Create the ceres problem and add the knots to the param list

            // Create the ceres problem
            ceres::Problem problem;
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;

            // vector<SE3d> pose0(Nlidar);
            // for(int lidx = 0; lidx < Nlidar; lidx++)
            //     pose0.push_back(localTraj[lidx]->pose((localTraj[lidx]->getMinTime() + localTraj[lidx]->maxTime())/2));

            // Create the problem
            CreateCeresProblem(problem, options, summary, localTraj, fixed_start, fixed_end);
            double cost_pose_init;
            double cost_pose_final;

            // Step 2.3: Add the lidar factors
            vector<ceres::internal::ResidualBlock *> res_ids_lidar;
            for(int lidx = 0; lidx < Nlidar; lidx++)
                for(int swIdx = 0; swIdx < WDZ; swIdx++)
                    AddLidarFactors(Coef[lidx][swIdx], localTraj[lidx], problem, res_ids_lidar);

            // Step 2.4 Add pose prior factors
            vector<ceres::internal::ResidualBlock *> res_ids_pose;
            for(int lidx = 0; lidx < Nlidar; lidx++)
                AddPosePriorFactors(localTraj[lidx], problem, res_ids_pose);

            // Step 2.5: Add motion prior factors
            vector<ceres::internal::ResidualBlock *> res_ids_gp;
            for(int lidx = 0; lidx < Nlidar; lidx++)
                AddMotionPriorFactors(localTraj[lidx], problem, res_ids_gp);

            // Step 2.6: Add relative extrinsic factors

            // Solve and visualize:
            ceres::Solve(options, &problem, &summary);

            // vector<SE3d> pose1(Nlidar);
            // for(int lidx = 0; lidx < Nlidar; lidx++)
            //     pose1.push_back(localTraj[lidx]->pose((localTraj[lidx]->getMinTime() + localTraj[lidx]->maxTime())/2));

            printf("LO Opt %d: J: %6.3f -> %6.3f. Iter: %d\n",
                    Nlidar,
                    summary.initial_cost, summary.final_cost, (int)(summary.iterations.size()));

            // Step 2.1: Copy the knots back to the global trajectory
            for (int lidx = 0; lidx < Nlidar; lidx++)
            {
                double tmin = localTraj[lidx]->getMinTime();

                // Find the starting knot in traj and copy to localtraj
                auto us = traj[lidx]->computeTimeIndex(tmin);
                int s = us.second;
                for(int kidx = 0; kidx < localTraj[lidx]->getNumKnots(); kidx++)
                    traj[lidx]->setKnot(kidx + s, localTraj[lidx]->getKnot(kidx));
            }

            // Step N: Visualize

            // Create the publishers ad hoc
            static vector<ros::Publisher> swTrajPub;
            if (swTrajPub.size() == 0)
                for(int lidx = 0; lidx < Nlidar; lidx++)
                    swTrajPub.push_back(nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", lidx), 1));

            // Sample and publish
            for(int lidx = 0; lidx < Nlidar; lidx++)
            {
                CloudPosePtr poseSampled = CloudPosePtr(new CloudPose());
                for (int swIdx = 0; swIdx < WDZ; swIdx++)
                {
                    double tb = swClouds[lidx][swIdx]->front().t;
                    double te = swClouds[lidx][swIdx]->back().t;
                    for(double ts = tb; ts < te; ts += 0.02)
                        if(localTraj[lidx]->TimeInInterval(ts))
                            poseSampled->points.push_back(myTf(localTraj[lidx]->pose(ts)).Pose6D(ts));
                }
                Util::publishCloud(swTrajPub[lidx], *poseSampled, ros::Time::now(), "world");
            }

        }
    }
};
