#include "unistd.h"
#include "ros/ros.h"

#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "sensor_msgs/PointCloud2.h"

#include "CloudMatcher.hpp"
#include "utility.h"

using namespace std;

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
                 const vector<vector<ros::Time>> &pcstamp,
                 CloudXYZIPtr &priormap,
                 double timestart,
                 vector<double> xyzypr_W_L0,
                 vector<CloudXYZIPtr> &pc0,
                 vector<myTf<double>> &tf_W_Li0)
{
    // Number of lidars
    int Nlidar = pcstamp.size();

    ROS_ASSERT(pc0.size() == Nlidar);
    ROS_ASSERT(tf_W_Li0.size() == Nlidar);

    // Find the init pose of each lidar
    for (int lidx = 0; lidx < Nlidar; lidx++)
    {
        // Merge the pointclouds in the first few seconds
        pc0[lidx] = CloudXYZIPtr(new CloudXYZI());
        int Ncloud = pcstamp[lidx].size();
        for(int cidx = 0; cidx < Ncloud; cidx++)
        {
            // Check if pointcloud is later
            if ((pcstamp[lidx][cidx] - pcstamp[lidx][0]).toSec() > timestart)
                break;

            // Merge lidar
            CloudXYZI temp; pcl::copyPointCloud(*clouds[lidx][cidx], temp);
            *pc0[lidx] += temp;

            // printf("P0 lidar %d, Cloud %d. Points: %d. Copied: %d\n", lidx, cidx, clouds[lidx][cidx]->size(), pc0[lidx]->size());
        }

        int Norg = pc0[lidx]->size();

        // Downsample the pointcloud
        pc0[lidx] = uniformDownsample<PointXYZI>(pc0[lidx], 0.1);
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

int main(int argc, char **argv)
{

    ros::init(argc, argv, "cartiber");
    ros::NodeHandle nh("~");
    boost::shared_ptr<ros::NodeHandle> nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Suppress warnings by pcl load

    printf("Lidar calibration.\n");

    // Get the dense prior map
    string priormap_file = "/home/tmn/ros_ws/dev_ws/src/cartiber/scripts/priormap.pcd";

    // Get the lidar bag file
    string lidar_bag_file = "/home/tmn/ros_ws/dev_ws/src/cartiber/scripts/cloud_noisy.bag";

    // Get the lidar topics
    vector<string> pc_topics = {"/lidar_0/points", "/lidar_1/points"};

    // Calculate the number of lidars
    int Nlidar = pc_topics.size();
    
    vector<double> xyzypr_W_L0(Nlidar*6, 0.0);
    if( nh.getParam("xyzypr_W_L0", xyzypr_W_L0) )
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

    double timestart = 3.0;

    // Load the priormap
    CloudXYZIPtr priormap(new CloudXYZI());
    pcl::io::loadPCDFile<PointXYZI>(priormap_file, *priormap);
    priormap = uniformDownsample<PointXYZI>(priormap, 0.05);

    // Converting the topic to index
    map<string, int> pctopicidx; for(int idx = 0; idx < pc_topics.size(); idx++) pctopicidx[pc_topics[idx]] = idx;

    // Storage of the pointclouds
    vector<vector<CloudXYZITPtr>> clouds(pc_topics.size());
    vector<vector<ros::Time>> pcstamp(pc_topics.size());

    // Load the bag file
    rosbag::Bag lidar_bag; lidar_bag.open(lidar_bag_file);
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
                PointOuster &pi = cloud_raw->points[pidx];
                PointXYZIT &po = cloud->points[pidx];
                po.x = pi.x;
                po.y = pi.y;
                po.z = pi.z;
                po.intensity = pi.intensity;
                po.t = double(pi.t/1.0e9);
            }

            clouds[lidx].push_back(cloud);
            pcstamp[lidx].push_back(pcMsg->header.stamp);

            printf("Loading pointcloud at time : %f. Cloud total: %d. Cloud size: %d / %d\r",
                    pcstamp[lidx].back().toSec(), clouds[lidx].size(), clouds[lidx].back()->size(), cloud_raw->size());
        }
    }

    vector<myTf<double>> tf_W_Li0(Nlidar);
    vector<CloudXYZIPtr> pc0(Nlidar);

    // Initialize the pose
    getInitPose(clouds, pcstamp, priormap, timestart, xyzypr_W_L0, pc0, tf_W_Li0);

    for(int lidx = 0; lidx < Nlidar; lidx++)
        pcl::transformPointCloud(*pc0[lidx], *pc0[lidx], tf_W_Li0[lidx].cast<float>().tfMat());

    ros::Rate rate(1);
    while(ros::ok())
    {
        ros::Time currTime = ros::Time::now();

        // Publish the pointclouds for visualization
        static ros::Publisher pmpub = nh.advertise<sensor_msgs::PointCloud2>("/priormap_viz", 1);
        Util::publishCloud(pmpub, *priormap, currTime, "world");

        static vector<ros::Publisher> scanInWPub;
        if(scanInWPub.size() == 0)
            for(int lidx = 0; lidx < Nlidar; lidx++)
                scanInWPub.push_back(nh.advertise<sensor_msgs::PointCloud2>(myprintf("/lidar_%d_inW", lidx), 1));

        for(int lidx = 0; lidx < Nlidar; lidx++)
            Util::publishCloud(scanInWPub[lidx], *pc0[lidx], currTime, "world");

        // Sleep
        rate.sleep();
    }
}