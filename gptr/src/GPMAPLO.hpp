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

// All about gaussian process
#include "GaussianProcess.hpp"

// Custom solver
#include "GNSolver.h"

using NodeHandlePtr = boost::shared_ptr<ros::NodeHandle>;

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

    double min_planarity = 0.2;
    double max_plane_dis = 0.3;

    // How many point clouds to import into the sliding window
    int WINDOW_SIZE = 10;

    // Switch to use ceres
    bool use_ceres = false;

    // My custom Gauss Newton solver
    GNSolverPtr mySolver;

    // Outer iterations
    int max_gniter = 3;

    // Initial pose of the lidars
    SE3d T_W_Li0;

    // Spline to model the trajectory of each lidar
    GaussianProcessPtr traj;

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
    double mpSigGa = 10;
    double mpSigNu = 10;
    double smSigmaR = 10;
    double smSigmaP = 10;

    // Time to fix the knot
    double fixed_start = 0.0;
    double fixed_end = 0.0;

    // Buffer for the pointcloud segments
    mutex cloud_seg_buf_mtx;
    deque<CloudXYZITPtr> cloud_seg_buf;

    // Publisher
    ros::Publisher trajPub;
    ros::Publisher swTrajPub;
    ros::Publisher assocCloudPub;
    ros::Publisher deskewedCloudPub;

public:

    // Destructor
   ~GPMAPLO() {};

    GPMAPLO(NodeHandlePtr &nh_ptr_, mutex & nh_mtx, const SE3d &T_W_Li0_, double t0, int &LIDX_)
        : nh_ptr(nh_ptr_), T_W_Li0(T_W_Li0_), LIDX(LIDX_)
    {
        lock_guard<mutex> lg(nh_mtx);

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
        nh_ptr->getParam("mpSigGa", mpSigGa);
        nh_ptr->getParam("mpSigNu", mpSigNu);
        // Weight for the smoothness factor
        nh_ptr->getParam("smSigmaR", smSigmaR);
        nh_ptr->getParam("smSigmaP", smSigmaP);

        // Association params
        nh_ptr->getParam("min_planarity", min_planarity);
        nh_ptr->getParam("max_plane_dis", max_plane_dis);

        trajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/gp_traj", LIDX), 1);
        swTrajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
        assocCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
        deskewedCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/cloud_inW", LIDX), 1);
        
        // Create the solver
        mySolver = GNSolverPtr(new GNSolver(nh_ptr, LIDX));

        // Some report to confirm the params are set
        printf("Window size: %2d. Fixes: %.3f, %.3f. DK: %.3f, %2d. lidar_weight: %.3f. ppSigma: %.3f, %.3f. mpSigGa: %.3f, %.3f\n",
                WINDOW_SIZE, fixed_start, fixed_end, tshift, DK, lidar_weight, ppSigmaR, ppSigmaP, mpSigGa, mpSigNu);
        // printf("GPMAPLO subscribing to %s\n\n", cloudSegTopic.c_str());

        Matrix3d SigGa = Vector3d(mpSigGa, mpSigGa, mpSigGa).asDiagonal();
        Matrix3d SigNu = Vector3d(mpSigNu, mpSigNu, mpSigNu).asDiagonal();

        traj = GaussianProcessPtr(new GaussianProcess(deltaT, SigGa, SigNu, true));
        traj->setStartTime(t0);
        traj->setKnot(0, GPState(t0, T_W_Li0));

    }

    // void CloudSegmentCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    // {
    //     CloudXYZITPtr cloudSeg(new CloudXYZIT());
    //     pcl::fromROSMsg(*msg, *cloudSeg);
    //     cloud_seg_buf_mtx.lock();
    //     cloud_seg_buf.push_back(cloudSeg);
    //     cloud_seg_buf_mtx.unlock();
    // }

    void AddCloudSeg(CloudXYZITPtr &msg)
    {
        lock_guard<mutex> lg(cloud_seg_buf_mtx);

        // Copy the pointcloud to local memory
        CloudXYZITPtr cloudSeg(new CloudXYZIT()); *cloudSeg = *msg;
        // Store on the cloud segment buffer
        cloud_seg_buf.push_back(cloudSeg);

        // printf("GPMAPLO %d rec cloud. Time: %f. Size: %d.\n", LIDX, cloudSeg->points.front().t, cloud_seg_buf.size());
    }

    void Associate(GaussianProcessPtr &traj, const KdFLANNPtr &kdtreeMap, const CloudXYZIPtr &priormap,
                   const CloudXYZITPtr &cloudRaw, const CloudXYZIPtr &cloudInB, const CloudXYZIPtr &cloudInW,
                   vector<LidarCoef> &Coef)
    {
        ROS_ASSERT_MSG(cloudRaw->size() == cloudInB->size(),
                       "cloudRaw: %d. cloudInB: %d", cloudRaw->size(), cloudInB->size());

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

                if (!traj->TimeInInterval(tpoint, 1e-6))
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
                if(Util::fitPlane(nbrPoints, min_planarity, max_plane_dis, Coef_[pidx].n, Coef_[pidx].plnrty))
                {
                    // ROS_ASSERT(tpoint >= 0);
                    Coef_[pidx].t = tpoint;
                    Coef_[pidx].f = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
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

        if (Npoints == 0)
            return;

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

    void Visualize(double tmin, double tmax, deque<vector<LidarCoef>> &swCloudCoef, CloudXYZIPtr &cloudUndiInW, bool publish_full_traj=false)
    {
        if (publish_full_traj)
        {
            CloudPosePtr trajCP = CloudPosePtr(new CloudPose());
            for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
            {
                trajCP->points.push_back(myTf(traj->getKnotPose(kidx)).Pose6D(traj->getKnotTime(kidx)));
                trajCP->points.back().intensity = (tmax - trajCP->points.back().t) < 0.1 ? 1.0 : 0.0;
            }

            // Publish global trajectory
            Util::publishCloud(trajPub, *trajCP, ros::Time::now(), "world");
        }

        // Sample and publish the slinding window trajectory
        CloudPosePtr poseSampled = CloudPosePtr(new CloudPose());
        for(double ts = tmin; ts < tmax; ts += traj->getDt()/5)
            if(traj->TimeInInterval(ts))
                poseSampled->points.push_back(myTf(traj->pose(ts)).Pose6D(ts));

        // static ros::Publisher swTrajPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/sw_opt", LIDX), 1);
        Util::publishCloud(swTrajPub, *poseSampled, ros::Time::now(), "world");

        CloudXYZIPtr assoc_cloud(new CloudXYZI());
        for (int widx = 0; widx < swCloudCoef.size(); widx++)
        {
            for(auto &coef : swCloudCoef[widx])
                {
                    PointXYZI p;
                    p.x = coef.finW.x();
                    p.y = coef.finW.y();
                    p.z = coef.finW.z();
                    p.intensity = widx;
                    assoc_cloud->push_back(p);
                }
        }
        
        // static ros::Publisher assocCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/assoc_cloud", LIDX), 1);
        if (assoc_cloud->size() != 0)
            Util::publishCloud(assocCloudPub, *assoc_cloud, ros::Time::now(), "world");

        // Publish the deskewed pointCloud
        Util::publishCloud(deskewedCloudPub, *cloudUndiInW, ros::Time::now(), "world");
    }

    void FindTraj(const KdFLANNPtr &kdTreeMap, const CloudXYZIPtr &priormap, double t0)
    {
        deque<CloudXYZITPtr> swCloudSeg;
        deque<CloudXYZIPtr > swCloudSegUndi;
        deque<CloudXYZIPtr > swCloudSegUndiInW;
        deque<vector<LidarCoef>> swCloudCoef;

        double timeout = -1;
        // Check the buffer
        while(ros::ok())
        {
            // Step 0: Poll and Extract the cloud segment -------------------------------------------------------------
            
            // Exit this thread if no data has been sent for more than 5 seconds
            if(timeout > 0 && ros::Time::now().toSec() - timeout > 2.0)
            {
                printf(KGRN "GPMAPLO %d completed.\n" RESET, LIDX);
                break;
            }

            CloudXYZITPtr cloudSeg;
            if(cloud_seg_buf.size() != 0)
            {
                std::lock_guard<mutex> lg(cloud_seg_buf_mtx);
                cloudSeg = cloud_seg_buf.front();
                cloud_seg_buf.pop_front();

                timeout = ros::Time::now().toSec();
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

            // No need to deskew for the first interval
            pcl::copyPointCloud(*swCloudSeg.back(), *swCloudSegUndi.back());

            // Transform cloud to the world frame for association
            SE3d pose = traj->pose(TSWEND);
            pcl::transformPointCloud(*swCloudSegUndi.back(), *swCloudSegUndiInW.back(), pose.translation(), pose.so3().unit_quaternion());

            // Step 2.1: Deskew the cloud segment
            Deskew(traj, swCloudSeg.back(), swCloudSegUndi.back());

            // Step 2.2: Associate the last pointcloud with the map
            Associate(traj, kdTreeMap, priormap, swCloudSeg.back(), swCloudSegUndi.back(), swCloudSegUndiInW.back(), swCloudCoef.back());          

            // Step 2.3: Create a local trajectory for optimization
            GaussianProcessPtr swTraj(new GaussianProcess(deltaT, traj->getSigGa(), traj->getSigNu()));
            int    umin = traj->computeTimeIndex(max(traj->getMinTime(), swCloudSeg.front()->points.front().t)).first;
            double tmin = traj->getKnotTime(umin);
            double tmax = min(traj->getMaxTime(), TSWEND);
            // Copy the knots {umin, umin+1, ...} from traj to swTraj
            for(int kidx = umin; kidx < traj->getNumKnots(); kidx++)
                swTraj->extendOneKnot(traj->getKnot(kidx));
            // Reset the start time
            swTraj->setStartTime(tmin);
            // Effective length of the sliding window
            int WDZ = min(int(swCloudSeg.size()), WINDOW_SIZE);

            tt_preopt.Toc();



            // Step 3: iterative optimization -------------------------------------------------------------------------

            // Sample the state before optimization 
            GPState Xts0 = traj->getStateAt(tmin);
            GPState Xte0 = traj->getStateAt(tmax);

            vector<string> report(max_gniter);  // A report to provide info on the internal of the optimization

            int optnum = -1; optnum++;
            int gniter = 0;
            while(gniter < max_gniter && traj->getMaxTime() > SKIPPED_TIME)
            {
                TicToc tt_solve;

                // Check for the next base for marginalization
                deque<int> swAbsKidx;
                for(int kidx = umin; kidx < traj->getNumKnots(); kidx++)
                    swAbsKidx.push_back(kidx);
                int swNextBaseKnot = -1;
                if (swCloudSeg.size() >= WINDOW_SIZE)
                    swNextBaseKnot = traj->computeTimeIndex(swCloudSeg[1]->points.front().t).first;

                // Solve
                mySolver->Solve(swTraj, swCloudCoef, gniter, swAbsKidx, swNextBaseKnot);

                // Get the report
                GNSolverReport gnreport = mySolver->GetReport();
                // Calculate the cost
                double J0 = gnreport.J0prior + gnreport.J0lidar + gnreport.J0mp2k;
                double JK = gnreport.JKprior + gnreport.JKlidar + gnreport.JKmp2k;
                JK = JK < 0 ? -1 : JK;

                // Get the covariance
                SparseMatrix<double> InvCov_ = mySolver->GetInvCov();
                MatrixXd InvCov = InvCov_.toDense();

                tt_solve.Toc();
                
                TicToc tt_aftop;
                // Step X: Copy the knots on the sliding window back to the global trajectory
                {
                    for(int kidx = 0; kidx < swTraj->getNumKnots(); kidx++)
                    {
                        double tgb = traj->getKnotTime(kidx + umin);
                        double tlc = swTraj->getKnotTime(kidx);
                        double ter = fabs(tlc - tgb);
                        ROS_ASSERT_MSG(ter < 1e-3, "Knot Time: %f, %f. Diff: %f.\n", tlc, tgb, ter);
                        traj->setKnot(kidx + umin, swTraj->getKnot(kidx));
                        traj->setKnotCovariance(kidx + umin, InvCov.block<STATE_DIM, STATE_DIM>(kidx*STATE_DIM, kidx*STATE_DIM));
                    }
                }

                // Deskew the point cloud and make new association
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
                    Associate(traj, kdTreeMap, priormap, swCloudSeg[widx], swCloudSegUndi[widx], swCloudSegUndiInW[widx], swCloudCoef[widx]);
                }

                // Visualize the result on the sliding window
                Visualize(tmin, tmax, swCloudCoef, swCloudSegUndiInW.back());

                tt_aftop.Toc();

                // Make report
                gniter++;
                // Print a report
                GPState XtsK = traj->getStateAt(tmin);
                GPState XteK = traj->getStateAt(tmax);
                double swTs = swCloudSeg.front()->points.front().t;
                double swTe = swCloudSeg.back()->points.back().t;
                report[gniter-1] = 
                myprintf("%sGPMAP%dLO#%d. OItr: %2d / %2d. GNItr: %2d. Umin: %4d. TKnot: %6.3f -> %6.3f. TCloud: %6.3f -> %6.3f.\n"
                         "Tprop: %2.0f. Tslv: %2.0f. Taftop: %2.0f. Tlp: %3.0f.\n"
                         "Factors: Lidar: %4d. Prior: %4d. Motion prior: %4d. Knots: %d / %d.\n"
                         "J0: %12.3f. Ldr: %9.3f. Prior: %9.3f. MP: %f.\n"
                         "JK: %12.3f. Ldr: %9.3f. Prior: %9.3f. MP: %f.\n"
                         "Poss0: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f. Acc: %6.3f, %6.3f, %6.3f. YPR: %6.3f, %6.3f, %6.3f.\n"
                         "PossK: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f. Acc: %6.3f, %6.3f, %6.3f. YPR: %6.3f, %6.3f, %6.3f.\n"
                         "Pose0: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f. Acc: %6.3f, %6.3f, %6.3f. YPR: %6.3f, %6.3f, %6.3f.\n"
                         "PoseK: %6.3f, %6.3f, %6.3f. Vel: %6.3f, %6.3f, %6.3f. Acc: %6.3f, %6.3f, %6.3f. YPR: %6.3f, %6.3f, %6.3f.\n"
                         RESET,
                         gniter == max_gniter ? KGRN : "", LIDX, optnum,
                         gniter, max_gniter, 1, umin, tmin, tmax, swTs, swTe,
                         tt_preopt.GetLastStop(), tt_solve.GetLastStop(), tt_aftop.GetLastStop(),
                         gniter == max_gniter ? tt_loop.Toc() : -1.0,
                         gnreport.lidarFactors, gnreport.priorFactors, gnreport.mp2kFactors, swTraj->getNumKnots(), traj->getNumKnots(),
                         J0, gnreport.J0lidar, gnreport.J0prior, gnreport.J0mp2k,
                         JK, gnreport.JKlidar, gnreport.JKprior, gnreport.JKmp2k,
                         Xts0.P.x(), Xts0.P.y(), Xts0.P.z(), Xts0.V.x(), Xts0.V.y(), Xts0.V.z(), Xts0.A.x(), Xts0.A.y(), Xts0.A.z(), Xts0.yaw(), Xts0.pitch(), Xts0.roll(),
                         XtsK.P.x(), XtsK.P.y(), XtsK.P.z(), XtsK.V.x(), XtsK.V.y(), XtsK.V.z(), XtsK.A.x(), XtsK.A.y(), XtsK.A.z(), XtsK.yaw(), XtsK.pitch(), XtsK.roll(),
                         Xte0.P.x(), Xte0.P.y(), Xte0.P.z(), Xte0.V.x(), Xte0.V.y(), Xte0.V.z(), Xte0.A.x(), Xte0.A.y(), Xte0.A.z(), Xte0.yaw(), Xte0.pitch(), Xte0.roll(),
                         XteK.P.x(), XteK.P.y(), XteK.P.z(), XteK.V.x(), XteK.V.y(), XteK.V.z(), XteK.A.x(), XteK.A.y(), XteK.A.z(), XteK.yaw(), XteK.pitch(), XteK.roll());
            }

            // Print the report
            if(traj->getMaxTime() > SKIPPED_TIME)
            {
                for(string &rep : report)
                    cout << rep;
                cout << endl;
            }

            // Step 4: Shift the sliding window if length exceeds threshold
            if (swCloudSeg.size() >= WINDOW_SIZE)
            {
                swCloudSeg.pop_front();
                swCloudSegUndi.pop_front();
                swCloudSegUndiInW.pop_front();
                swCloudCoef.pop_front();
            }
        }
    }

    GaussianProcessPtr &GetTraj()
    {
        return traj;
    }
};
