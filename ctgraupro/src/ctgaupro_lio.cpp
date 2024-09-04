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

// Dense prior map
CloudXYZIPtr priormap(new CloudXYZI());

// Get the lidar bag file
string lidar_bag_file = "";

// Number of clouds to work with
int MAX_CLOUDS = -1;

// Time to skip the estimation
double SKIPPED_TIME = 0.0;

// Get the imu topics
vector<string> imu_topic = {""};

// Get the lidar topics
vector<string> lidar_topic = {"/lidar_0/points"};

// Get the lidar type
vector<string> lidar_type = {"ouster"};

// Get the lidar stamp time (start /  end)
vector<string> stamp_time = {"start"};

// Check for log and load the GP trajectory with the control points in log
vector<int> resume_from_log = {0};

// Get the prior map leaf size
double pmap_leaf_size = 0.15;
vector<double> cloud_ds;

// Kdtree for priormap
KdFLANNPtr kdTreeMap(new KdFLANN());

// Spline knot length
double deltaT = 0.01;

// ikdtree of the priormap
ikdtreePtr ikdtPM;

// Number of poses per knot in the extrinsic optimization
int SW_CLOUDNUM = 20;
int SW_CLOUDSTEP = 2;
bool VIZ_ONLY = false;

double t_shift = 0.0;
int max_outer_iter = 3;
int max_inner_iter = 2;


vector<myTf<double>> T_B_Li_gndtr;

string log_dir = "/home/tmn/logs";

// Mutex for the node handle
mutex nh_mtx;

// Define the posespline
typedef std::shared_ptr<GPMAPLO> GPMAPLOPtr;
typedef std::shared_ptr<GaussianProcess> GaussianProcessPtr;

// vector<GPKFLOPtr> gpkflo;
vector<GPMAPLOPtr> gpmaplo;

template <typename Scalar = double, int RowSize = Dynamic, int ColSize = Dynamic>
Matrix<Scalar, RowSize, ColSize> load_dlm(const std::string &path, string dlm, int r_start = 0, int col_start = 0)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int row_idx = -1;
    int rows = 0;
    while (std::getline(indata, line))
    {
        row_idx++;
        if (row_idx < r_start)
            continue;

        std::stringstream lineStream(line);
        std::string cell;
        int col_idx = -1;
        while (std::getline(lineStream, cell, dlm[0]))
        {
            if (cell == dlm || cell.size() == 0)
                continue;

            col_idx++;
            if (col_idx < col_start)
                continue;

            values.push_back(std::stod(cell));
        }

        rows++;
    }

    return Map<const Matrix<Scalar, RowSize, ColSize, RowMajor>>(values.data(), rows, values.size() / rows);
}

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

void getInitPose(int lidx,
                 const vector<vector<CloudXYZITPtr>> &clouds,
                 const vector<vector<ros::Time>> &cloudstamp,
                 CloudXYZIPtr &priormap,
                 vector<double> &timestart,
                 const vector<double> &xyzypr_W_L0,
                 vector<CloudXYZIPtr> &pc0,
                 vector<myTf<double>> &tf_W_Li0)
{
    // Number of lidars
    int Nlidar = cloudstamp.size();

    // Time period to merge initial clouds
    double startup_merge_time = 3.0;

    ROS_ASSERT(pc0.size() == Nlidar);
    ROS_ASSERT(tf_W_Li0.size() == Nlidar);

    // // Find the init pose of each lidar
    // for (int lidx = 0; lidx < Nlidar; lidx++)
    // {
        // Merge the pointclouds in the first few seconds
        pc0[lidx] = CloudXYZIPtr(new CloudXYZI());
        int Ncloud = cloudstamp[lidx].size();
        for(int cidx = 0; cidx < Ncloud; cidx++)
        {
            // Check if pointcloud is later
            if ((cloudstamp[lidx][cidx] - cloudstamp[lidx][0]).toSec() > startup_merge_time)
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
        printf("Intial cloud of lidar %d, Points: %d -> %d\n", lidx, Norg, pc0[lidx]->size());

        // Find ICP alignment and refine
        CloudMatcher cm(0.1, 0.1);

        // Set the original position of the anchors
        Vector3d p_W_L0(xyzypr_W_L0[lidx*6 + 0], xyzypr_W_L0[lidx*6 + 1], xyzypr_W_L0[lidx*6 + 2]);
        Quaternd q_W_L0 = Util::YPR2Quat(xyzypr_W_L0[lidx*6 + 3], xyzypr_W_L0[lidx*6 + 4], xyzypr_W_L0[lidx*6 + 5]);
        myTf tf_W_L0(q_W_L0, p_W_L0);

        // // Find ICP pose
        // Matrix4f tfm_W_Li0;
        // double   icpFitness   = 0;
        // double   icpTime      = 0;
        // bool     icpconverged = cm.CheckICP(priormap, pc0[lidx], tf_W_L0.cast<float>().tfMat(), tfm_W_Li0, 0.2, 10, 1.0, icpFitness, icpTime);
        
        // tf_W_L0 = myTf(tfm_W_Li0);
        // printf("Lidar %d initial pose. %s. Time: %f. Fn: %f. XYZ: %f, %f, %f. YPR: %f, %f, %f.\n",
        //         lidx, icpconverged ? "Conv" : "Not Conv", icpTime, icpFitness,
        //         tf_W_L0.pos.x(), tf_W_L0.pos.y(), tf_W_L0.pos.z(),
        //         tf_W_L0.yaw(), tf_W_L0.pitch(), tf_W_L0.roll());

        // Find the refined pose
        IOAOptions ioaOpt;
        ioaOpt.init_tf = tf_W_L0;
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
    // }

    return;
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

void VisualizeGndtr(vector<CloudPosePtr> &gndtrCloud)
{
    // Number of lidars
    int Nlidar = gndtrCloud.size();
    
    // Create the publisher
    ros::Publisher pmpub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap_viz", 1);
    ros::Publisher gndtrPub[Nlidar];
    for(int idx = 0; idx < Nlidar; idx++)
        gndtrPub[idx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/gndtr", idx), 10);

    // Publish gndtr every x seconds
    ros::Rate rate(10);
    while(ros::ok())
    {
        ros::Time currTime = ros::Time::now();

        // Publish the prior map for visualization
        static int count = 0;
        count++;
        if (count < 5)
            Util::publishCloud(pmpub, *priormap, currTime, "world");

        // Publish the grountruth
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            if(gndtrCloud[lidx]->size() == 0)
            {
                // printf(KYEL "GND pose is empty\n" RESET);
                continue;
            }

            // printf("Publish GND pose cloud of %d points\n", gndtrCloud[lidx]->size());
            Util::publishCloud(gndtrPub[lidx], *gndtrCloud[lidx], ros::Time::now(), "world");
        }

        // Sleep
        rate.sleep();
    }    
}

int main(int argc, char **argv)
{
    // Initalize ros nodes
    ros::init(argc, argv, "ctgaupro");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
    
    // Supress the pcl warning
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Suppress warnings by pcl load

    printf(KGRN "Multi-Lidar Coupled Motion Estimation Started.\n" RESET);
 
    /* #region Read parameters --------------------------------------------------------------------------------------*/

    // Knot length
    nh_ptr->getParam("deltaT", deltaT);
    printf("Gaussian process with knot length: %f\n", deltaT);

    // Get the user define parameters
    nh_ptr->getParam("priormap_file", priormap_file);
    nh_ptr->getParam("lidar_bag_file", lidar_bag_file);
    
    nh_ptr->getParam("MAX_CLOUDS", MAX_CLOUDS);
    nh_ptr->getParam("SKIPPED_TIME", SKIPPED_TIME);
    
    nh_ptr->getParam("imu_topic", imu_topic);
    nh_ptr->getParam("lidar_topic", lidar_topic);
    nh_ptr->getParam("lidar_type", lidar_type);
    nh_ptr->getParam("stamp_time", stamp_time);
    nh_ptr->getParam("resume_from_log", resume_from_log);

    // Determine the number of lidar
    int Nlidar = lidar_topic.size();
    int Nimu = imu_topic.size();

    // Get the leaf size for prior map
    nh_ptr->getParam("pmap_leaf_size", pmap_leaf_size);

    // Get the leaf size for lidar pointclouds
    cloud_ds = vector<double>(Nlidar, 0.1);
    nh_ptr->getParam("cloud_ds", cloud_ds);

    // Find the settings for cross trajectory optimmization
    VIZ_ONLY = Util::GetBoolParam(nh_ptr, "VIZ_ONLY", false);
    nh_ptr->getParam("SW_CLOUDNUM", SW_CLOUDNUM);
    nh_ptr->getParam("SW_CLOUDSTEP", SW_CLOUDSTEP);
    nh_ptr->getParam("t_shift", t_shift);
    nh_ptr->getParam("max_outer_iter", max_outer_iter);
    nh_ptr->getParam("max_inner_iter", max_inner_iter);

    // Location to save the logs
    nh_ptr->getParam("log_dir", log_dir);

    // Some notifications
    printf("Get bag at %s and prior map at %s.\n", lidar_bag_file.c_str(), priormap_file.c_str());
    printf("Lidar info: \n");
    for(int lidx = 0; lidx < Nlidar; lidx++)
        printf("Type: %s.\tDs: %f. Topic %s.\n", lidar_type[lidx].c_str(), cloud_ds[lidx], lidar_topic[lidx].c_str());
    printf("Maximum number of clouds: %d\n", MAX_CLOUDS);

    printf("IMU info: \n");
    int imuCount = 0;
    for(int iidx = 0; iidx < Nimu; iidx++)
        printf("Topic %s.\n", imu_topic[iidx].c_str());

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
    vector<double> xtrz_gndtr(Nlidar*6, 0.0);
    if( nh_ptr->getParam("xtrz_gndtr", xtrz_gndtr) )
    {
        if (xtrz_gndtr.size() < Nlidar*6)
        {
            printf(KYEL "xtrz_gndtr missing values. Setting all to zeros \n" RESET);
            xtrz_gndtr = vector<double>(Nlidar*6, 0.0);
        }
        else
        {
            printf("xtrz_gndtr found: \n");
            for(int i = 0; i < Nlidar; i++)
            {
                T_B_Li_gndtr[i] = myTf(Util::YPR2Quat(xtrz_gndtr[i*6 + 3], xtrz_gndtr[i*6 + 4], xtrz_gndtr[i*6 + 5]),
                                             Vector3d(xtrz_gndtr[i*6 + 0], xtrz_gndtr[i*6 + 1], xtrz_gndtr[i*6 + 2]));

                for(int j = 0; j < 6; j++)
                    printf("%f, ", xtrz_gndtr[i*6 + j]);
                cout << endl;
            }
        }
    }
    else
    {
        printf("Failed to get xyzypr_W_L0. Setting all to zeros\n");
        xyzypr_W_L0 = vector<double>(Nlidar*6, 0.0);
    }

    /* #endregion Read parameters -----------------------------------------------------------------------------------*/
 
    /* #region Load the priormap ------------------------------------------------------------------------------------*/

    pcl::io::loadPCDFile<PointXYZI>(priormap_file, *priormap);
    priormap = uniformDownsample<PointXYZI>(priormap, pmap_leaf_size);
    // Create the kd tree
    printf("Building the prior map");
    kdTreeMap->setInputCloud(priormap);

    /* #endregion Load the priormap ---------------------------------------------------------------------------------*/
 
    /* #region Load the data ----------------------------------------------------------------------------------------*/

    map<string, int> imutopicidx;
    for(int iidx = 0; iidx < Nimu; iidx++)
        imutopicidx[imu_topic[iidx]] = iidx;

    // Converting the topics to index
    map<string, int> pctopicidx;
    for(int lidx = 0; lidx < Nlidar; lidx++)
        pctopicidx[lidar_topic[lidx]] = lidx;

    // Storage of the pointclouds
    vector<vector<RosImuPtr>> imus(Nimu);
    vector<vector<CloudXYZITPtr>> clouds(Nlidar);
    vector<vector<ros::Time>> cloudstamp(Nlidar);
    vector<tf2_msgs::TFMessage> gndtr;

    vector<string> queried_topics = lidar_topic;
    for(auto &topic : imu_topic)
        queried_topics.push_back(topic);
    queried_topics.push_back("/tf");

    // Load the bag file
    rosbag::Bag lidar_bag;
    lidar_bag.open(lidar_bag_file);
    rosbag::View view(lidar_bag, rosbag::TopicQuery(queried_topics));

    // Load the message
    for (rosbag::MessageInstance const m : view)
    {
        sensor_msgs::PointCloud2::ConstPtr pcMsgOuster = m.instantiate<sensor_msgs::PointCloud2>();
        livox_ros_driver::CustomMsg::ConstPtr pcMsgLivox = m.instantiate<livox_ros_driver::CustomMsg>();
        tf2_msgs::TFMessage::ConstPtr gndtrMsg = m.instantiate<tf2_msgs::TFMessage>();
        RosImuPtr imuMsg = m.instantiate<sensor_msgs::Imu>();

        CloudXYZITPtr cloud(new CloudXYZIT());
        ros::Time stamp;

        string topic = m.getTopic();
        int lidx = pctopicidx.find(topic) == pctopicidx.end() ? -1 : pctopicidx[topic];
        int NpointRaw = 0;
        int NpointDS = 0;

        // Copy the ground truth
        if (gndtrMsg != nullptr && topic == "/tf")
        {
            tf2_msgs::TFMessage msg = *gndtrMsg;
            gndtr.push_back(msg);
            continue;
        }

        if (pcMsgOuster != nullptr && lidar_type[lidx] == "ouster")
        {
            CloudOusterPtr cloud_raw(new CloudOuster());
            pcl::fromROSMsg(*pcMsgOuster, *cloud_raw);

            // Find the time stamp
            double sweeptime = (cloud_raw->points.back().t - cloud_raw->points.front().t)/1e9;
            double timebase = stamp_time[lidx] == "start" ? pcMsgOuster->header.stamp.toSec() : pcMsgOuster->header.stamp.toSec() - sweeptime;
            stamp = ros::Time(timebase);

            NpointRaw = cloud_raw->size();

            // Downsample the pointcloud
            CloudOusterPtr cloud_raw_ds = uniformDownsample<PointOuster>(cloud_raw, cloud_ds[lidx]);

            auto copyPoint = [](PointOuster &pi, PointXYZIT &po, double timebase) -> void
            {
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.t = timebase + pi.t/1.0e9;
                po.intensity = pi.intensity;
            };

            cloud->resize(cloud_raw_ds->size()+2);
            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < cloud_raw_ds->size(); pidx++)
                copyPoint(cloud_raw_ds->points[pidx], cloud->points[pidx + 1], timebase);

            copyPoint(cloud_raw->points.front(), cloud->points.front(), timebase);
            copyPoint(cloud_raw->points.back(), cloud->points.back(), timebase);

            NpointDS = cloud->size();

        }
        else if (pcMsgLivox != nullptr && lidar_type[lidx] == "livox")
        {
            NpointRaw = pcMsgLivox->point_num;
            
            // Find the time stamp
            double sweeptime = (pcMsgLivox->points.back().offset_time - pcMsgLivox->points.front().offset_time)/1e9;
            double timebase = stamp_time[lidx] == "start" ? pcMsgLivox->header.stamp.toSec() : pcMsgLivox->header.stamp.toSec() - sweeptime;
            stamp = ros::Time(timebase);

            auto copyPoint = [](const livox_ros_driver::CustomPoint &pi, PointXYZIT &po, double timebase) -> void
            {
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.t = timebase + pi.offset_time/1.0e9;
                po.intensity = pi.reflectivity/255.0*1000;
            };

            CloudXYZITPtr cloud_temp(new CloudXYZIT());
            cloud_temp->resize(pcMsgLivox->point_num);
            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int pidx = 0; pidx < pcMsgLivox->point_num; pidx++)
                copyPoint(pcMsgLivox->points[pidx], cloud_temp->points[pidx], timebase);

            // Downsample
            CloudXYZITPtr cloud_temp_ds = uniformDownsample<PointXYZIT>(cloud_temp, cloud_ds[lidx]);
            
            // Copy data to final container
            cloud->resize(cloud_temp_ds->size()+2);

            #pragma omp parallel for num_threads(MAX_THREADS)
                for(int pidx = 0; pidx < cloud_temp_ds->size(); pidx++)
                    cloud->points[pidx + 1] = cloud_temp_ds->points[pidx];

            cloud->points.front() = cloud_temp->points.front();
            cloud->points.back() = cloud_temp->points.back();

            NpointDS = cloud->size();
        }

        // Extract the imu data
        int iidx = imutopicidx.find(topic) == imutopicidx.end() ? -1 : imutopicidx[topic];
        if(imuMsg != nullptr && iidx >= 0 && iidx < Nimu)
            imus[iidx].push_back(imuMsg);
        
        // Save the pointcloud if it has data
        if (cloud->size() != 0)
        {
            clouds[lidx].push_back(cloud);
            cloudstamp[lidx].push_back(stamp);

            printf("Loading pointcloud from lidar %d at time: %.3f, %.3f. Cloud total: %d. Cloud size: %d / %d. Topic: %s.\r",
                    lidx,
                    cloudstamp[lidx].back().toSec(),
                    clouds[lidx].back()->points.front().t,
                    clouds[lidx].size(), NpointRaw, NpointDS, lidar_topic[lidx].c_str());
            cout << endl;

            // Confirm the time correctness
            ROS_ASSERT_MSG(fabs(cloudstamp[lidx].back().toSec() - clouds[lidx].back()->points.front().t) < 1e-9,
                           "Time: %f, %f.",
                           cloudstamp[lidx].back().toSec(), clouds[lidx].back()->points.front().t);
        }

        // Check if pointcloud is sufficient
        if (MAX_CLOUDS > 0 && clouds.front().size() >= MAX_CLOUDS)
            break;
    }

    /* #endregion Load the data -------------------------------------------------------------------------------------*/
 
    /* #region Extract the ground truth and publish -----------------------------------------------------------------*/

    vector<vector<double>> gndtr_ts(Nlidar);
    vector<CloudPosePtr> gndtrCloud(Nlidar);
    for(auto &cloud : gndtrCloud)
    {
        cloud = CloudPosePtr(new CloudPose());
        cloud->clear();
    }
    
    // Add points to gndtr Cloud
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
            
            // Save the time stamps
            gndtr_ts[lidar_id].push_back(pose.t);
        }
    }
    for(auto &cloud : gndtrCloud)
        printf("GNDTR cloud size: %d point(s)\n");

    // Create thread for visualizing groundtruth
    thread vizGtr = thread(VisualizeGndtr, std::ref(gndtrCloud));


    /* #endregion Extract the ground truth and publish --------------------------------------------------------------*/
 
    /* #region Initialize the pose of each lidar --------------------------------------------------------------------*/
    
    // Initial coordinates of the lidar
    vector<myTf<double>> tf_W_Li0(Nlidar);
    vector<CloudXYZIPtr> pc0(Nlidar);

    vector<double> timestart(Nlidar);
    vector<thread> poseInitThread(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        poseInitThread[lidx] = thread(getInitPose, lidx, std::ref(clouds), std::ref(cloudstamp), std::ref(priormap),
                                      std::ref(timestart), std::ref(xyzypr_W_L0), std::ref(pc0), std::ref(tf_W_Li0));

    for(int lidx = 0; lidx < Nlidar; lidx++)
        poseInitThread[lidx].join();

    /* #endregion Initialize the pose of each lidar -----------------------------------------------------------------*/
 
    /* #region Split the pointcloud by time -------------------------------------------------------------------------*/
    
    vector<vector<CloudXYZITPtr>> cloudsx(Nlidar); cloudsx[0] = clouds[0];
    for(int lidx = 1; lidx < Nlidar; lidx++)
    {
        printf("Split cloud %d\n", lidx);
        syncLidar(clouds[0], clouds[lidx], cloudsx[lidx]);
    }

    /* #endregion Split the pointcloud by time ----------------------------------------------------------------------*/

    /* #region Split the imu by cloud time --------------------------------------------------------------------------*/

    vector<vector<ImuSequence>> imusx(Nimu);
    for(int iidx = 0; iidx < Nimu; iidx++)
    {
        printf("Split imu %d\n", iidx);
        imusx[iidx].resize(cloudsx[0].size());
        
        // #pragma omp parallel num_threads(MAX_THREADS)
        for(int isidx = 0; isidx < imus[iidx].size(); isidx++)
        {
            ImuSample imu(imus[iidx][isidx]);
            for(int cidx = 0; cidx < cloudsx[0].size(); cidx++)
            {                
                if(imu.t > cloudsx[0][cidx]->points.back().t)
                    continue;
                else if(cloudsx[0][cidx]->points.front().t <= imu.t && imu.t < cloudsx[0][cidx]->points.back().t)
                    imusx[iidx][cidx].push_back(imu);
            }
        }
    }

    // // Report the distribution
    // for(int iidx = 0; iidx < Nimu; iidx++)
    // {
    //     for(int cidx = 0; cidx < cloudsx[0].size(); cidx++)
    //     {
    //         printf("IMU %2d Sequence %4d, sample %3d. ImuItv: [%.3f %.3f]. CloudItv. [%.3f %.3f].\n",
    //                 iidx, cidx, imusx[iidx][cidx].size(),
    //                 imusx[iidx][cidx].startTime(), imusx[iidx][cidx].finalTime(),
    //                 cloudsx[0][cidx]->points.front().t, cloudsx[0][cidx]->points.back().t);
    //     }
    // }
    
    /* #endregion Split the imu by cloud time -----------------------------------------------------------------------*/
  
    /* #region Create the LOAM modules ------------------------------------------------------------------------------*/

    gpmaplo = vector<GPMAPLOPtr>(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        // Create the gpmaplo objects
        gpmaplo[lidx] = GPMAPLOPtr(new GPMAPLO(nh_ptr, nh_mtx, tf_W_Li0[lidx].getSE3(), cloudsx[lidx].front()->points.front().t, lidx));

    // If there is a log, load them up
    for(int lidx = 0; lidx < Nlidar; lidx++)
    {
        string log_file = log_dir + myprintf("/gptraj_%d.csv", lidx);
        if(resume_from_log[lidx] == 1 && file_exist(log_file))
        {
            printf("Loading traj file: %s\n", log_file.c_str());
            gpmaplo[lidx]->GetTraj()->loadTrajectory(log_file);

            // GaussianProcessPtr &traj = gpmaplo[lidx]->GetTraj();
            // for(int kidx = 0; kidx < traj->getNumKnots(); kidx++)
            // {
            //     GPState<double> x = traj->getKnot(kidx);
            //     Quaternd q = x.R.unit_quaternion();
            //     printf("Lidar %d. Knot: %d. XYZ: %9.3f, %9.3f, %9.3f. Q: %9.3f, %9.3f, %9.3f, %9.3f.\n",
            //             lidx, kidx, x.P.x(), x.P.y(), x.P.z(), q.x(), q.y(), q.z(), q.w());
            // }
        }
    }
 
    // Create the estimation module
    GPMLCPtr gpmlc(new GPMLC(nh_ptr, Nlidar));
    vector<GaussianProcessPtr> trajs;
    for(auto &lo : gpmaplo)
        trajs.push_back(lo->GetTraj());

    vector<vector<geometry_msgs::PoseStamped>> extrinsic_poses(Nlidar);
 
    /* #endregion Create the LOAM modules ---------------------------------------------------------------------------*/
 
    /* #region Do optimization with inter-trajectory factors --------------------------------------------------------*/

    for(int outer_iter = 0; outer_iter < max_outer_iter; outer_iter++)
    {
        int SW_CLOUDSTEP = 1;
        for(int cidx = outer_iter; cidx < cloudsx[0].size() - SW_CLOUDSTEP; cidx+= int(SW_CLOUDSTEP))
        {
            if ((cloudsx[0][cidx]->points.front().t - cloudsx.front().front()->points.front().t) < SKIPPED_TIME)
                continue;

            int SW_BEG = cidx;
            int SW_END = min(cidx + SW_CLOUDNUM, int(cloudsx[0].size())-1);
            int SW_MID = min(cidx + SW_CLOUDSTEP, int(cloudsx[0].size())-1);

            // The effective length of the sliding window by the number of point clouds
            int SW_CLOUDNUM_EFF = SW_END - SW_BEG;

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

            // Deskew, Associate, Estimate, repeat max_inner_iter times
            for(int inner_iter = 0; inner_iter < max_inner_iter; inner_iter++)
            {
                // Create buffers for lidar coefficients
                vector<deque<CloudXYZITPtr>> swCloud(Nlidar, deque<CloudXYZITPtr>(SW_CLOUDNUM_EFF));
                vector<deque<CloudXYZIPtr >> swCloudUndi(Nlidar, deque<CloudXYZIPtr>(SW_CLOUDNUM_EFF));
                vector<deque<CloudXYZIPtr >> swCloudUndiInW(Nlidar, deque<CloudXYZIPtr>(SW_CLOUDNUM_EFF));
                vector<deque<vector<LidarCoef>>> swCloudCoef(Nlidar, deque<vector<LidarCoef>>(SW_CLOUDNUM_EFF));

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

                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    for(int idx = SW_BEG; idx < SW_END; idx++)
                    {
                        int swIdx = idx - SW_BEG;
                        swCloud[lidx][swIdx] = uniformDownsample<PointXYZIT>(cloudsx[lidx][idx], cloud_ds[lidx]);
                        ProcessCloud(gpmaplo[lidx], swCloud[lidx][swIdx], swCloudUndi[lidx][swIdx], swCloudUndiInW[lidx][swIdx], swCloudCoef[lidx][swIdx]);
                    }
                }

                // Optimize
                if(!VIZ_ONLY)
                    gpmlc->Evaluate(inner_iter, outer_iter, trajs, tmin, tmax, tmid, swCloudCoef, inner_iter >= max_inner_iter - 1, T_B_Li_gndtr);

                for(int lidx = 0; lidx < Nlidar; lidx++)
                {
                    if (inner_iter == max_inner_iter - 1)
                    {
                        SE3d se3 = gpmlc->GetExtrinsics(lidx);
                        geometry_msgs::PoseStamped pose;
                        pose.header.stamp = ros::Time(tmax);
                        pose.header.frame_id = "lidar_0";
                        pose.pose.position.x = se3.translation().x();
                        pose.pose.position.y = se3.translation().y();
                        pose.pose.position.z = se3.translation().z();
                        pose.pose.orientation.x = se3.so3().unit_quaternion().x();
                        pose.pose.orientation.y = se3.so3().unit_quaternion().y();
                        pose.pose.orientation.z = se3.so3().unit_quaternion().z();
                        pose.pose.orientation.w = se3.so3().unit_quaternion().w();
                        extrinsic_poses[lidx].push_back(pose);
                    }
                }

                // Visualize the result on each trajectory
                {
                    static vector<ros::Publisher*> odomPub(Nlidar, nullptr);
                    static vector<ros::Publisher*> marker_pub(Nlidar, nullptr);
                    static vector<nav_msgs::Odometry> odomMsg(Nlidar, nav_msgs::Odometry());

                    for(int lidx = 0; lidx < Nlidar; lidx++)
                    {
                        gpmaplo[lidx]->Visualize(tmin, tmax, swCloudCoef[lidx], swCloudUndiInW[lidx].back(), true);

                        // Publish an odom topic for each lidar
                        if (odomPub[lidx] == nullptr)
                            odomPub[lidx] = new ros::Publisher(nh_ptr->advertise<nav_msgs::Odometry>(myprintf("/lidar_%d/odom", lidx), 1));

                        double ts = tmax - trajs[lidx]->getDt()/2;
                        SE3d pose = trajs[lidx]->pose(tmax);    
                        odomMsg[lidx].header.stamp = ros::Time(tmax);
                        odomMsg[lidx].header.frame_id = "world";
                        odomMsg[lidx].child_frame_id = myprintf("lidar_%d_body", lidx);
                        odomMsg[lidx].pose.pose.position.x = pose.translation().x();
                        odomMsg[lidx].pose.pose.position.y = pose.translation().y();
                        odomMsg[lidx].pose.pose.position.z = pose.translation().z();
                        odomMsg[lidx].pose.pose.orientation.x = pose.unit_quaternion().x();
                        odomMsg[lidx].pose.pose.orientation.y = pose.unit_quaternion().y();
                        odomMsg[lidx].pose.pose.orientation.z = pose.unit_quaternion().z();
                        odomMsg[lidx].pose.pose.orientation.w = pose.unit_quaternion().w();
                        odomPub[lidx]->publish(odomMsg[lidx]);

                        if (lidx == 0)
                            continue;

                        // printf("Lidar %d. Pos: %f, %f, %f\n", lidx, pose.translation().x(), pose.translation().y(), pose.translation().z());
                        if(marker_pub[lidx] == nullptr)
                            marker_pub[lidx] = new ros::Publisher(nh_ptr->advertise<visualization_msgs::Marker>(myprintf("/lidar_%d/extr_marker", lidx), 1));

                        // Publish a line between the lidars
                        visualization_msgs::Marker line_strip;
                        line_strip.header.frame_id = "world";
                        line_strip.header.stamp = ros::Time::now();
                        line_strip.ns = "lines";
                        line_strip.action = visualization_msgs::Marker::ADD;
                        line_strip.pose.orientation.w = 1.0;
                        line_strip.id = 0;
                        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
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
                        p.x = odomMsg[lidx].pose.pose.position.x;;
                        p.y = odomMsg[lidx].pose.pose.position.y;;
                        p.z = odomMsg[lidx].pose.pose.position.z;;
                        line_strip.points.push_back(p);
                        marker_pub[lidx]->publish(line_strip);
                    }
                }
            }
        }

        // Reset the marginalization factor
        gpmlc->Reset();

        // Log the result

        // Create directories if they do not exist
        string output_dir = log_dir + myprintf("/run_%02d/", outer_iter);
        std::filesystem::create_directories(output_dir);

        // Save the trajectory and estimation result
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            // string log_file = log_dir + myprintf("/gptraj_%d.csv", lidx);
            printf("Exporting trajectory logs to %s.\n", output_dir.c_str());
            gpmaplo[lidx]->GetTraj()->saveTrajectory(output_dir, lidx, gndtr_ts[lidx]);
        }

        // Log the extrinsics
        string xts_log = output_dir + "/extrinsics_" + std::to_string(1) + ".csv";
        std::ofstream xts_logfile;
        xts_logfile.open(xts_log); // Open the file for writing
        xts_logfile.precision(std::numeric_limits<double>::digits10 + 1);
        xts_logfile << "t, x, y, z, qx, qy, qz, qw" << endl;
        for(auto &pose : extrinsic_poses[1])
        {
            xts_logfile << pose.header.stamp.toSec() << ","
                        << pose.pose.position.x << ","
                        << pose.pose.position.y << ","
                        << pose.pose.position.z << ","
                        << pose.pose.orientation.x << ","
                        << pose.pose.orientation.y << ","
                        << pose.pose.orientation.z << ","
                        << pose.pose.orientation.w << "," << endl;
        }
        xts_logfile.close();
    }

    /* #endregion Do optimization with inter-trajectory factors -----------------------------------------------------*/
 
    /* #region Create the pose sampling publisher -------------------------------------------------------------------*/

    vector<ros::Publisher> poseSamplePub(Nlidar);
    for(int lidx = 0; lidx < Nlidar; lidx++)
        poseSamplePub[lidx] = nh_ptr->advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d/pose_sampled", lidx), 1);

    // Loop in waiting
    ros::Rate rate(0.2);
    while(ros::ok())
        rate.sleep();

    /* #endregion Create the pose sampling publisher ----------------------------------------------------------------*/
 
}