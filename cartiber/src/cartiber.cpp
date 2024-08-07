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
#include "GPMLC.h"

// Factor for optimization
// #include "factor/PoseAnalyticFactor.h"
// #include "factor/ExtrinsicFactor.h"
// #include "factor/FullExtrinsicFactor.h"
#include "factor/GPExtrinsicFactor.h"
#include "factor/GPMotionPriorTwoKnotsFactor.h"
// #include "factor/GPMotionPriorTwoKnotsFactorTMN.hpp"
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

// Number of poses per knot in the extrinsic optimization
int CLOUDS_SW = 2;
int Nseg = 1;
double t_shift = 0.0;
int max_outer_iter = 3;
int max_inner_iter = 2;

vector<myTf<double>> T_B_Li_gndtr;

// Mutex for the node handle
mutex nh_mtx;

// Define the posespline
typedef std::shared_ptr<GPMAPLO> GPMAPLOPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

// vector<GPKFLOPtr> gpkflo;
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
    nh_ptr->getParam("deltaT", deltaT);

    printf("SPLINE order %d with knot length: %f\n", SPLINE_N, deltaT);

    // Get the user define parameters
    nh_ptr->getParam("priormap_file", priormap_file);
    nh_ptr->getParam("lidar_bag_file", lidar_bag_file);
    nh_ptr->getParam("MAX_CLOUDS", MAX_CLOUDS);
    nh_ptr->getParam("pc_topics", pc_topics);
    nh_ptr->getParam("lidar_type", lidar_type);
    
    double SKIPPED_TIME = 0.0;
    nh_ptr->getParam("SKIPPED_TIME", SKIPPED_TIME);

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

    // Find the settings for cross trajectory optimmization
    nh_ptr->getParam("CLOUDS_SW", CLOUDS_SW);
    nh_ptr->getParam("Nseg", Nseg);
    nh_ptr->getParam("t_shift", t_shift);
    nh_ptr->getParam("max_outer_iter", max_outer_iter);
    nh_ptr->getParam("max_inner_iter", max_inner_iter);

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

    T_B_Li_gndtr.resize(Nlidar);
    vector<double> xtrns_gndtr(Nlidar*6, 0.0);
    if( nh_ptr->getParam("xtrns_gndtr", xtrns_gndtr) )
    {
        if (xtrns_gndtr.size() < Nlidar*6)
        {
            printf(KYEL "xtrns_gndtr missing values. Setting all to zeros \n" RESET);
            xtrns_gndtr = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            printf("xtrns_gndtr found: \n");
            for(int i = 0; i < Nlidar; i++)
            {
                T_B_Li_gndtr[i] = myTf(Util::YPR2Quat(xtrns_gndtr[i*6 + 3], xtrns_gndtr[i*6 + 4], xtrns_gndtr[i*6 + 5]),
                                             Vector3d(xtrns_gndtr[i*6 + 0], xtrns_gndtr[i*6 + 1], xtrns_gndtr[i*6 + 2]));

                for(int j = 0; j < 6; j++)
                    printf("%f, ", xtrns_gndtr[i*6 + j]);
                cout << endl;
            }
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


    // Split the pointcloud by time.
    vector<vector<CloudXYZITPtr>> cloudsx(Nlidar); cloudsx[0] = clouds[0];
    for(int lidx = 1; lidx < Nlidar; lidx++)
    {
        printf("Split cloud %d\n", lidx);
        syncLidar(clouds[0], clouds[lidx], cloudsx[lidx]);
    }

    // Create the LOAM modules
    gpmaplo = vector<GPMAPLOPtr>(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        // Create the gpmaplo objects
        gpmaplo[lidx] = GPMAPLOPtr(new GPMAPLO(nh_ptr, nh_mtx, tf_W_Li0[lidx].getSE3(), cloudsx[lidx].front()->points.front().t, lidx));

    // Create the estimation module
    GPMLCPtr gpmlc(new GPMLC(nh_ptr));
    vector<GaussianProcessPtr> trajs = {gpmaplo[0]->GetTraj(), gpmaplo[1]->GetTraj()};

    // Do optimization with inter-trajectory factors
    for(int outer_iter = 0; outer_iter < max_outer_iter; outer_iter++)
    {
        int CLOUDS_SW_HALF = int(CLOUDS_SW/2);
        for(int cidx = outer_iter; cidx < cloudsx[0].size() - CLOUDS_SW_HALF; cidx+= int(CLOUDS_SW_HALF))
        {
            if (cloudsx[0][cidx]->points.front().t < SKIPPED_TIME)
                continue;

            int SW_BEG = cidx;
            int SW_END = min(cidx + CLOUDS_SW, int(cloudsx[0].size())-1);
            int SW_MID = min(cidx + CLOUDS_SW_HALF, int(cloudsx[0].size())-1);

            // The effective length of the sliding window by the number of point clouds
            int CLOUDS_SW_EFF = SW_END - SW_BEG;

            double tmin = cloudsx[0][SW_BEG]->points.front().t;     // Start time of the sliding window
            double tmax = cloudsx[0][SW_END]->points.back().t;      // End time of the sliding window
            double tmid = cloudsx[0][SW_MID]->points.front().t;     // Next start time of the sliding window,
                                                                    // also determines the marginalization time limit
            
            // Extend the trajectories
            for(int lidx = 0; lidx < Nlidar; lidx++)
            {
                while(trajs[lidx]->getMaxTime() < tmax)
                    trajs[lidx]->extendOneKnot();
            }
            
            // Deskew, Associate, Estimate, repeat three times
            for(int inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
            {
                // Create buffers for lidar coefficients
                vector<deque<vector<LidarCoef>>> swCloudCoef(2, deque<vector<LidarCoef>>(CLOUDS_SW_EFF));

                deque<CloudXYZITPtr> swCloud0(CLOUDS_SW_EFF);
                deque<CloudXYZIPtr> swCloudUndi0(CLOUDS_SW_EFF);
                deque<CloudXYZIPtr> swCloudUndiInW0(CLOUDS_SW_EFF);
                deque<vector<LidarCoef>> &swCloudCoef0 = swCloudCoef[0];

                deque<CloudXYZITPtr> swCloudi(CLOUDS_SW_EFF);
                deque<CloudXYZIPtr> swCloudUndii(CLOUDS_SW_EFF);
                deque<CloudXYZIPtr> swCloudUndiInWi(CLOUDS_SW_EFF);
                deque<vector<LidarCoef>> &swCloudCoefi = swCloudCoef[1];

                // Deskew, Transform and Associate
                auto ProcessCloud = [&kdTreeMap, &priormap](GPMAPLOPtr &gpmaplo, CloudXYZITPtr &cloudRaw, CloudXYZIPtr &cloudUndi,
                                                            CloudXYZIPtr &cloudUndiInW, vector<LidarCoef> &cloudCoeff) -> void
                {
                    GaussianProcessPtr traj = gpmaplo->GetTraj();
                    
                    // Deskew
                    cloudUndi = CloudXYZIPtr(new CloudXYZI());
                    gpmaplo->Deskew(traj, cloudRaw, cloudUndi);
                    
                    // Transform
                    cloudUndiInW = CloudXYZIPtr(new CloudXYZI());
                    SE3d pose = traj->pose(cloudRaw->points.back().t);
                    pcl::transformPointCloud(*cloudUndi, *cloudUndiInW, pose.translation(), pose.so3().unit_quaternion());

                    // Associate
                    gpmaplo->Associate(traj, kdTreeMap, priormap, cloudRaw, cloudUndi, cloudUndiInW, cloudCoeff);

                };
                
                for(int idx = SW_BEG; idx < SW_END; idx++)
                {
                    int swIdx = idx - SW_BEG;

                    swCloud0[swIdx] = uniformDownsample<PointXYZIT>(cloudsx[0][idx], 0.1);
                    swCloudi[swIdx] = uniformDownsample<PointXYZIT>(cloudsx[1][idx], 0.1);

                    ProcessCloud(gpmaplo[0], swCloud0[swIdx], swCloudUndi0[swIdx], swCloudUndiInW0[swIdx], swCloudCoef0[swIdx]);
                    ProcessCloud(gpmaplo[1], swCloudi[swIdx], swCloudUndii[swIdx], swCloudUndiInWi[swIdx], swCloudCoefi[swIdx]);
                }

                // Optimize
                gpmlc->Evaluate(outer_iter, trajs, tmin, tmax, tmid, swCloudCoef, inner_iter < max_inner_iter - 1, T_B_Li_gndtr[1]);

                // Visualize the result on each trajectory
                {
                    gpmaplo[0]->Visualize(tmin, tmax, swCloudCoef0, swCloudUndiInW0.back(), true);
                    gpmaplo[1]->Visualize(tmin, tmax, swCloudCoefi, swCloudUndiInWi.back(), true);

                    // Publish an odom topic for each lidar
                    static vector<ros::Publisher> odomPub = {nh_ptr->advertise<nav_msgs::Odometry>(myprintf("/lidar_%d/odom", 0), 1),
                                                             nh_ptr->advertise<nav_msgs::Odometry>(myprintf("/lidar_%d/odom", 1), 1)};
                    static vector<nav_msgs::Odometry> odomMsg(2);
                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        double ts = trajs[lidx]->getMaxTime() - trajs[lidx]->getDt()/2;
                        SE3d pose = trajs[lidx]->pose(ts);
                        odomMsg[lidx].header.stamp = ros::Time(ts);
                        odomMsg[lidx].header.frame_id = "world";
                        odomMsg[lidx].child_frame_id = myprintf("lidar_%d_body", lidx);
                        odomMsg[lidx].pose.pose.position.x = pose.translation().x();
                        odomMsg[lidx].pose.pose.position.y = pose.translation().y();
                        odomMsg[lidx].pose.pose.position.z = pose.translation().z();
                        odomMsg[lidx].pose.pose.orientation.x = pose.unit_quaternion().x();
                        odomMsg[lidx].pose.pose.orientation.y = pose.unit_quaternion().y();
                        odomMsg[lidx].pose.pose.orientation.z = pose.unit_quaternion().z();
                        odomMsg[lidx].pose.pose.orientation.w = pose.unit_quaternion().w();
                        odomPub[lidx].publish(odomMsg[lidx]);
                    }

                    // Publish a line between the lidars
                    static ros::Publisher marker_pub = nh_ptr->advertise<visualization_msgs::Marker>("extr_marker", 1);
                    visualization_msgs::Marker line_strip;
                    line_strip.header.frame_id = "world";
                    line_strip.header.stamp = ros::Time::now();
                    line_strip.ns = "lines";
                    line_strip.action = visualization_msgs::Marker::ADD;
                    line_strip.pose.orientation.w = 1.0;
                    line_strip.id = 0;
                    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
                    // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
                    line_strip.scale.x = 0.05;
                    // Line strip is red
                    line_strip.color.r = 0.0;
                    line_strip.color.g = 0.0;
                    line_strip.color.b = 1.0;
                    line_strip.color.a = 1.0;
                    // Create the vertices for the points and lines
                    geometry_msgs::Point p;
                    p.x = odomMsg[0].pose.pose.position.x;
                    p.y = odomMsg[0].pose.pose.position.y;
                    p.z = odomMsg[0].pose.pose.position.z;
                    line_strip.points.push_back(p);
                    p.x = odomMsg[1].pose.pose.position.x;;
                    p.y = odomMsg[1].pose.pose.position.y;;
                    p.z = odomMsg[1].pose.pose.position.z;;
                    line_strip.points.push_back(p);
                    marker_pub.publish(line_strip);
                }
            }
        }

        // Reset the marginalization factor
        gpmlc->Reset();
    }

    // Create the pose sampling publisher
    vector<ros::Publisher> poseSamplePub(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        poseSamplePub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/pose_sampled", lidx), 1);

    // Loop in waiting
    ros::Rate rate(0.2);
    while(ros::ok())
        rate.sleep();
}