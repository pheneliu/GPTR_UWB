#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utility.h"

#include "factor/PoseAnalyticFactor.h"
#include "factor/GaussianPriorFactor.h"
#include "factor/PointToPlaneAnalyticFactor.hpp"

using NodeHandlePtr = boost::shared_ptr<ros::NodeHandle>;
using PoseSplinePtr = std::shared_ptr<PoseSplineX>;

class SGPLO
{
private:

    NodeHandlePtr nh_ptr;

    // How many point clouds to import into the sliding window
    int WINDOW_SIZE = 10;

    // Extrinsics of the lidars
    vector<SE3d> T_L0_Li;

    // Spline to model the trajectory of each lidar
    vector<PoseSplinePtr> traj;

    // Associate params
    int knnSize = 6;
    double minKnnSqDis = 0.5*0.5;
    double minKnnNbrDis = 0.1;

    // Rate of skipping the factors
    int lidar_ds_rate = 1;

    int SPLINE_N = 4;
    double deltaT = 0.1;

    double wR = 10;
    double wP = 10;

    // Time to fix the knot
    double fixed_start = 0.0;
    double fixed_end = 0.0;

    // Param for creating huber loss
    double lidar_loss_thres = 1.0;

public:

    // Destructor
   ~SGPLO() {};

    SGPLO(NodeHandlePtr &nh_ptr_, vector<PoseSplinePtr> &traj_, vector<SE3d> &T_L0_Li_)
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

        nh_ptr->getParam("wR", wR);
        nh_ptr->getParam("wP", wP);

        printf("Window size: %d. Fixes: <%f, >%f. Weight: %f, %f\n", WINDOW_SIZE, fixed_start, fixed_end, wR, wP);
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

    void Deskew(PoseSplinePtr &traj, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudDeskewedInB)
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
                            vector<PoseSplinePtr> &localTraj, double fixed_start, double fixed_end)
    {
        int Nlidar = traj.size();
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = MAX_THREADS;
        options.max_num_iterations = 50;

        ceres::LocalParameterization *local_parameterization = new basalt::LieAnalyticLocalParameterization<Sophus::SO3d>();
        for (int lidx = 0; lidx < Nlidar; lidx++)
        {
            int KNOTS = localTraj[lidx]->numKnots();
            
            // Add the parameter blocks for rotation
            for (int kidx = 0; kidx < KNOTS; kidx++)
                problem.AddParameterBlock(localTraj[lidx]->getKnotSO3(kidx).data(), 4, local_parameterization);

            // Add the parameter blocks for position
            for (int kidx = 0; kidx < KNOTS; kidx++)
                problem.AddParameterBlock(localTraj[lidx]->getKnotPos(kidx).data(), 3);

            // Fix the knots
            if (fixed_start >= 0)
                for (int kidx = 0; kidx < KNOTS; kidx++)
                {
                    if (localTraj[lidx]->getKnotTime(kidx) <= localTraj[lidx]->minTime() + fixed_start)
                    {
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(kidx).data());
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(kidx).data());
                    }
                }

            if (fixed_end >= 0)
            {
                for (int kidx = 0; kidx < KNOTS; kidx++)
                {
                    if (localTraj[lidx]->getKnotTime(kidx) >= localTraj[lidx]->maxTime() - fixed_end)
                    {
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(KNOTS - 1 - kidx).data());
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(KNOTS - 1 - kidx).data());
                    }
                }
            }    
        }
    }

    bool TimeInInterval(PoseSplinePtr &traj, double t, double eps = 0)
    {
        return (t >= traj->minTime() + eps && t <= traj->maxTime() - eps);
    }

    void AddLidarFactors(vector<LidarCoef> &coef, PoseSplinePtr &traj, ceres::Problem &problem,
                         vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        // map<int, int> coupled_knots;

        static int skip = -1;
        for (auto &coef : coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!TimeInInterval(traj, coef.t, 1e-6))
                continue;

            skip++;
            if (skip % lidar_ds_rate != 0)
                continue;
            
            auto us = traj->computeTIndex(coef.t);
            int base_knot = us.second;
            double s = us.first;

            vector<double *> factor_param_blocks;

            // Add the parameter blocks for rotation
            for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotSO3(knot_idx).data());

            // Add the parameter blocks for position
            for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotPos(knot_idx).data());

            ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
            ceres::CostFunction *cost_function = new PointToPlaneAnalyticFactor(coef.finW, coef.f, coef.n, coef.plnrty, SPLINE_N, traj->getDt(), s);
            auto res = problem.AddResidualBlock(cost_function, lidar_loss_function, factor_param_blocks);
            res_ids_lidar.push_back(res);
        }
    }

    void AddPosePriorFactors(PoseSplinePtr &traj, ceres::Problem &problem,
                             vector<ceres::internal::ResidualBlock *> &res_ids_pose)
    {
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

        // Add the pose factors with priors sampled from previous spline
        for (double t = traj->minTime(); t < traj->maxTime(); t+=0.05)
        {
            // Continue if sample is in the window
            if (!TimeInInterval(traj, t, 1e-6))
                continue;

            auto   us = traj->computeTIndex(t);
            int    base_knot = us.second;
            double s = us.first;

            vector<double *> factor_param_blocks;

            // Find the coupled poses
            for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotSO3(knot_idx).data());

            for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotPos(knot_idx).data());

            ceres::CostFunction *cost_function = new PoseAnalyticFactor(traj->pose(t), 10.0, 10.0, SPLINE_N, traj->getDt(), s);
            auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
            res_ids_pose.push_back(res_block);
        }
    }

    void AddExtrinsicPoseFactors(PoseSplinePtr &traj, ceres::Problem &problem,
                                 vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        // Add the extrinsic factors at fix intervals
        for (double t = traj->minTime(); t < traj->maxTime() - 0.05; t += 0.05)
        {
            
        }
    }

    void AddGaussianPriorFactors(PoseSplinePtr &traj, ceres::Problem &problem,
                                 vector<ceres::internal::ResidualBlock *> &res_ids_gp)
    {
        int DK = 1;
        double tshift = 0.5*(traj->getDt());

        // Add the GP factors based on knot difference
        for (int kidx = 0; kidx < traj->numKnots() - DK; kidx++)
        {

            double ta = traj->getKnotTime(kidx) + tshift;
            double tb = traj->getKnotTime(kidx + DK) + tshift;

            if (!TimeInInterval(traj, ta, 1e-6) || !TimeInInterval(traj, tb, 1e-6))
                continue;

            // Find the coupled control points
            auto   usa   = traj->computeTIndex(ta);
            int    basea = usa.second;
            double sa    = usa.first;

            auto   usb   = traj->computeTIndex(tb);
            int    baseb = usb.second;
            double sb    = usb.first;

            // Confirm that basea and baseb are DK knots apart
            ROS_ASSERT(baseb - basea == DK);

            vector<double *> factor_param_blocks;

            // Add the parameter blocks for rotation
            for (int knot_idx = basea; knot_idx < basea + SPLINE_N + DK; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotSO3(knot_idx).data());

            // Add the parameter blocks for position
            for (int knot_idx = basea; knot_idx < basea + SPLINE_N + DK; knot_idx++)
                factor_param_blocks.emplace_back(traj->getKnotPos(knot_idx).data());

            // Create the factor
            double gp_loss_thres = -1;
            ceres::LossFunction *gp_loss_func = gp_loss_thres == -1 ? NULL : new ceres::HuberLoss(lidar_loss_thres);
            ceres::CostFunction *cost_function = new GaussianPriorFactor(wR, wP, SPLINE_N, traj->getDt(), sa, sb, tb - ta, DK);

            auto res_block = problem.AddResidualBlock(cost_function, gp_loss_func, factor_param_blocks);
            res_ids_gp.push_back(res_block);

            return;
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
            vector<PoseSplinePtr> localTraj(Nlidar);
            for (int lidx = 0; lidx < Nlidar; lidx++)
            {
                double tmin = max(traj[lidx]->minTime(), swClouds[lidx].front()->points.front().t);
                double tmax = min(traj[lidx]->maxTime(), swClouds[lidx].back()->points.back().t);

                // Find the knots related to this trajectory
                localTraj[lidx] = PoseSplinePtr(new PoseSplineX(SPLINE_N, deltaT));
                localTraj[lidx]->setStartTime(tmin);
                localTraj[lidx]->extendKnotsTo(tmax, SE3d());

                // Find the starting knot in traj and copy to localtraj
                auto us = traj[lidx]->computeTIndex(tmin);
                int s = us.second;
                for(int kidx = 0; kidx < localTraj[lidx]->numKnots(); kidx++)
                    localTraj[lidx]->setKnot(traj[lidx]->getKnot(kidx + s), kidx);
            }

            // Step 2,2: Create the ceres problem and add the knots to the param list
            
            // Create the ceres problem
            ceres::Problem problem;
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;

            // vector<SE3d> pose0(Nlidar);
            // for(int lidx = 0; lidx < Nlidar; lidx++)
            //     pose0.push_back(localTraj[lidx]->pose((localTraj[lidx]->minTime() + localTraj[lidx]->maxTime())/2));

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

            // Step 2.5: Add gaussian prior factors
            vector<ceres::internal::ResidualBlock *> res_ids_gp;
            for(int lidx = 0; lidx < Nlidar; lidx++)
                AddGaussianPriorFactors(localTraj[lidx], problem, res_ids_gp);

            // Step 2.6: Add relative extrinsic factors

            // Solve and visualize:
            ceres::Solve(options, &problem, &summary);

            // vector<SE3d> pose1(Nlidar);
            // for(int lidx = 0; lidx < Nlidar; lidx++)
            //     pose1.push_back(localTraj[lidx]->pose((localTraj[lidx]->minTime() + localTraj[lidx]->maxTime())/2));

            printf("LO Opt %d: J: %6.3f -> %6.3f. Iter: %d\n",
                    Nlidar,
                    summary.initial_cost, summary.final_cost, (int)(summary.iterations.size()));

            // Step 2.1: Copy the knots back to the global trajectory
            for (int lidx = 0; lidx < Nlidar; lidx++)
            {
                double tmin = localTraj[lidx]->minTime();

                // Find the starting knot in traj and copy to localtraj
                auto us = traj[lidx]->computeTIndex(tmin);
                int s = us.second;
                for(int kidx = 0; kidx < localTraj[lidx]->numKnots(); kidx++)
                    traj[lidx]->setKnot(localTraj[lidx]->getKnot(kidx), kidx + s);
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
                        if(TimeInInterval(localTraj[lidx], ts))
                            poseSampled->points.push_back(myTf(localTraj[lidx]->pose(ts)).Pose6D(ts));
                }
                Util::publishCloud(swTrajPub[lidx], *poseSampled, ros::Time::now(), "world");
            }
        }
    }
};
