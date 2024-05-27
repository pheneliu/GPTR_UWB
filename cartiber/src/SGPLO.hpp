#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utility.h"

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
    }

    void Associate(const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        if (priormap->size() > knnSize)
        {
            int pointsCount = cloudInW->points.size();
            vector<LidarCoef> Coef_;
            Coef_.resize(pointsCount);
            
            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int pidx = 0; pidx < pointsCount; pidx++)
            {
                PointXYZI pointInB = cloudInB->points[pidx];
                PointXYZI pointInW = cloudInW->points[pidx];

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
                        nbrPoints.push_back(priormap->points[knn_idx[idx]]);
                else
                    continue;

                // Fit the plane
                if(Util::fitPlane(nbrPoints, 0.5, 0.1, Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    Coef_[pidx].t = 0;
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
        cloudDeskewedInB->clear();
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
                    if (localTraj[lidx]->getKnotTime(kidx) <= fixed_start)
                    {
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(kidx).data());
                        problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(kidx).data());
                    }
                }

            if (fixed_end >= 0.0)
            {
                for (int kidx = 0; kidx < SPLINE_N; kidx++)
                {
                    problem.SetParameterBlockConstant(localTraj[lidx]->getKnotSO3(KNOTS - 1 - kidx).data());
                    problem.SetParameterBlockConstant(localTraj[lidx]->getKnotPos(KNOTS - 1 - kidx).data());
                }
            }    
        }
    }

    bool PointInInterval(PoseSplinePtr &traj, double t, double eps)
    {
        return (t >= traj->minTime() + eps || t <= traj->maxTime() - eps);
    }

    void AddLidarFactors(vector<LidarCoef> &coef, PoseSplinePtr &traj, ceres::Problem &problem,
                         vector<ceres::internal::ResidualBlock *> &res_ids_lidar)
    {
        static int skip = -1;
        for (auto &coef : coef)
        {
            // Skip if lidar coef is not assigned
            if (coef.t < 0)
                continue;

            if (!PointInInterval(traj, coef.t, 1e-6))
                continue;

            skip++;
            if (skip % lidar_ds_rate != 0)
                continue;
            
            auto us = traj->computeTIndex(coef.t);
            int base_knot = us.second;
            double s = us.first;

            // // Add dataset
            // PointXYZIT point;
            // point.x = coef.finW.x();
            // point.y = coef.finW.y();
            // point.z = coef.finW.z();
            // point.t = coef.t;
            // assocCloud->push_back(point);

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
                    swClouds[lidx].push_back(clouds[lidx][gbIdx]);

            // Step 0: Deskew
            for (int lidx = 0; lidx < Nlidar; lidx++)
                for (int swIdx = 0; swIdx < WDZ; swIdx++)
                {
                    double te = swClouds[lidx][swIdx]->back().t;

                    // Deskew to the end time of the scan
                    Deskew(traj[lidx], swClouds[lidx][swIdx], swCloudsDeskewed[lidx][swIdx]);
                    
                    // Transform pointcloud to the world frame
                    myTf tf_W_Be(traj[lidx]->pose(te));
                    pcl::transformPointCloud(*swCloudsDeskewed[lidx][swIdx],
                                             *swCloudsDeskewedInW[lidx][swIdx],
                                              tf_W_Be.pos, tf_W_Be.rot);
                }

            // Step 1: Associate
            vector<vector<LidarCoef>> Coef;
            for (int lidx = 0; lidx < Nlidar; lidx++)
                for (int swIdx = 0; swIdx < WDZ; swIdx++)
                    Associate(kdTreeMap, priormap,
                              swCloudsDeskewed[lidx][swIdx],
                              swCloudsDeskewedInW[lidx][swIdx],
                              Coef[lidx]);

            // Step 2: Build the optimization problem
            
            // Step 2.1: Copy the knots to the local trajectories
            vector<PoseSplinePtr> localTraj;
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
            
            // Create the problem
            CreateCeresProblem(problem, options, summary, localTraj, fixed_start, fixed_end);

            // Step 2.3: Add the lidar factors
            vector<ceres::internal::ResidualBlock *> res_ids_lidar;
            for(int lidx = 0; lidx < Nlidar; lidx++)
                AddLidarFactors(Coef[lidx], traj[lidx], problem, res_ids_lidar);

            // Step 2.4: Add gaussian prior factors

            // Step 2.5: Add relative extrinsic factors
        }
    }
};
