#include "unistd.h"

// PCL utilities
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

// ROS utilities
#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "sensor_msgs/PointCloud2.h"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Custom built utilities
#include "CloudMatcher.hpp"
#include "GPLO.hpp"
#include "utility.h"

using namespace std;

// Node handle
boost::shared_ptr<ros::NodeHandle> nh_ptr;

// Get the dense prior map
string priormap_file = "";

// Get the lidar bag file
string lidar_bag_file = "";

// Get the lidar topics
vector<string> pc_topics = {"/lidar_0/points"};

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
using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
// typedef Matrix<double, 12, 12> MatrixNd;

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

void DeskewBySpline(CloudXYZITPtr &cloudin, CloudXYZITPtr &cloudout, PoseSplinePtr &traj)
{
    int Npoints = cloudin->size();
    cloudout->resize(Npoints);
    
    // double t0 = cloudin->points[0].t;
    // myTf T_W_Bk(traj->pose(t0));

    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int pidx = 0; pidx < Npoints; pidx++)
    {        
        PointXYZIT &pi = cloudin->points[pidx];
        PointXYZIT &po = cloudout->points[pidx];
        
        double ts = pi.t;
        myTf T_W_Bs = traj->pose(ts);

        Vector3d pi_inBs(pi.x, pi.y, pi.z);
        Vector3d pi_inW = T_W_Bs.rot*pi_inBs + T_W_Bs.pos;

        po.x = pi_inW.x();
        po.y = pi_inW.y();
        po.z = pi_inW.z();
        po.t = pi.t;
        po.intensity = pi.intensity;
    }
}

void trajectoryEstimate(const CloudXYZIPtr &priormap, const ikdtreePtr &ikdtPM,
                        const vector<CloudXYZITPtr> &clouds, const vector<ros::Time> &cloudstamp,
                        const myTf<double> &tf_W_L0, int lidx, mutex &nh_mtx)
{
    // Calculate the trajectory by sim  ple iekf first, before refining with spline
    StateWithCov Xhat0(cloudstamp.front().toSec(), tf_W_L0.rot, tf_W_L0.pos, Vector3d(0, 0, 0), Vector3d(0, 0, 0), 1.0);
    GPLO gplo(lidx, Xhat0, UW_NOISE, UV_NOISE, 0.5*0.5, 0.1, nh_ptr, nh_mtx);
    gplo.FindTraj(kdTreeMap, priormap, clouds, cloudstamp);

    // // Number of clouds
    // int Ncloud = clouds.size();
    
    // // Construct a spline
    // PoseSplinePtr lidarTraj = nullptr;
    // lidarTraj = PoseSplinePtr(new PoseSplineX(SPLINE_N, deltaT));
    // lidarTraj->setStartTime(cloudstamp.front().toSec());
    // lidarTraj->extendKnotsTo(clouds.back()->points.back().t, tf_W_L0.getSE3());

    // // Extract the pointclouds and fit it to the sliding window
    // for(int cidx = 0; cidx < Ncloud; cidx+=sw_shift)
    // {
    //     int sw_startidx = cidx;
    //     int sw_finalidx = min(Ncloud, cidx + sw_shift);

    //     // Extract the pointclouds on the sliding window, deskew, then convert them to the world frame
    //     vector<CloudXYZITPtr> swCloudInW(sw_finalidx - sw_startidx);
    //     for(int cidx = sw_startidx; cidx < sw_finalidx; cidx++)
    //     {
    //         int swcidx = cidx - sw_startidx;
    //         double tstart = clouds[swcidx]->points.front().t;
    //         swCloudInW[swcidx] = CloudXYZITPtr(new CloudXYZIT());
    //         swCloudInW[swcidx]->resize(clouds[cidx]->size());

    //         // Deskew the pointcloud using the spline
    //         DeskewBySpline(clouds[cidx], swCloudInW[swcidx], lidarTraj);

    //         // pcl::transformPointCloud(*clouds[cidx], *swCloudInW[swcidx],
    //         //                           myTf<double>(lidarTraj->pose(tstart)).cast<float>().tfMat());
    //     }
    // }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cartiber");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Suppress warnings by pcl load

    printf("Lidar calibration started.\n");

    // Get the user define parameters
    nh_ptr->getParam("priormap_file", priormap_file);
    nh_ptr->getParam("lidar_bag_file", lidar_bag_file);
    nh_ptr->getParam("pc_topics", pc_topics);
    printf("Get bag at %s and prior map at %s\n", lidar_bag_file.c_str(), priormap_file.c_str());
    printf("Lidar topics: \n");
    for(auto topic : pc_topics)
        cout << topic << endl;

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
        sensor_msgs::PointCloud2::ConstPtr pcMsg = m.instantiate<sensor_msgs::PointCloud2>();
        if (pcMsg != nullptr)
        {
            int lidx = pctopicidx[m.getTopic()];
            
            CloudOusterPtr cloud_raw(new CloudOuster());
            pcl::fromROSMsg(*pcMsg, *cloud_raw);

            CloudXYZITPtr cloud(new CloudXYZIT()); cloud->resize(cloud_raw->size());
            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < cloud_raw->size(); pidx++)
            {
                double pt0 = pcMsg->header.stamp.toSec();
                PointOuster &pi = cloud_raw->points[pidx];
                PointXYZIT &po = cloud->points[pidx];
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.intensity = pi.intensity;
                po.t = pt0 + pi.t/1.0e9;
            }

            clouds[lidx].push_back(cloud);
            cloudstamp[lidx].push_back(pcMsg->header.stamp);

            printf("Loading pointcloud from lidar %d at time: %.3f, %.3f. Cloud total: %d. Cloud size: %d / %d\r",
                    lidx,
                    cloudstamp[lidx].back().toSec(),
                    clouds[lidx].back()->points.front().t,
                    clouds[lidx].size(), clouds[lidx].back()->size(), cloud_raw->size());
            cout << endl;

            // // Confirm the time correctness
            ROS_ASSERT_MSG(cloudstamp[lidx].back().toSec() == clouds[lidx].back()->points.front().t,
                           "Time: %f, %f.",
                           cloudstamp[lidx].back().toSec(), clouds[lidx].back()->points.front().t);
        }
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
    vector<GPLO> gplo;
    vector<thread> trajEst;
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Calculate the trajectory by sim  ple iekf first, before refining with spline
        StateWithCov Xhat0(cloudstamp[lidx].front().toSec(), tf_W_Li0[lidx].rot, tf_W_Li0[lidx].pos, Vector3d(0, 0, 0), Vector3d(0, 0, 0), 1.0);
        gplo.push_back(GPLO(lidx, Xhat0, UW_NOISE, UV_NOISE, 0.5*0.5, 0.1, nh_ptr, nh_mtx));
        trajEst.push_back(thread(std::bind(&GPLO::FindTraj, &gplo[lidx],
                                 std::ref(kdTreeMap), std::ref(priormap),
                                 std::ref(clouds[lidx]), std::ref(cloudstamp[lidx]))));
    }

    // Wait for the trajectory estimate to finish
    for(int lidx = 0; lidx < Nlidar; lidx++)
        trajEst[lidx].join();

    ros::Rate rate(1);
    while(ros::ok())
    {
        ros::Time currTime = ros::Time::now();

        // Publish the prior map for visualization
        Util::publishCloud(pmpub, *priormap, currTime, "world");

        // static vector<ros::Publisher> scanInWPub;
        // if(scanInWPub.size() == 0)
        //     for(int lidx = 0; lidx < Nlidar; lidx++)
        //         scanInWPub.push_back(nh.advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d_inW", lidx), 1));

        // for(int lidx = 0; lidx < Nlidar; lidx++)
        // {
        //     CloudXYZIPtr temp(new CloudXYZI());
        //     pcl::transformPointCloud(*pc0[lidx], *temp, tf_W_Li0[lidx].cast<float>().tfMat());
        //     Util::publishCloud(scanInWPub[lidx], *temp, currTime, "world");

        //     // Pose prior calculated from iekf
        //     Util::publishCloud(pppub[lidx], *poseprior[lidx], currTime, "world");
        // }

        // Sleep
        rate.sleep();
    }
}