# Presiquite

* Please build [SFUISE](https://github.com/KIT-ISAS/SFUISE) in your workspace to have the cf_msg, which is required in ctgaupro.
* Please install Ceres 2.1.0 is needed.

# Testing with lidar:

Some dependencies are needed to compile the package

To compile the package for use with livox lidars (avia, mid-70, mid 360), you need to install [Livox ROS driver (forked)](https://github.com/brytsknguyen/livox_ros_driver) and [Livox ROS driver2 (forked)](https://github.com/brytsknguyen/livox_ros_driver2) (you need to install [LIVOX-SDK](https://github.com/Livox-SDK/Livox-SDK) and [LIVOX-SDK2](https://github.com/Livox-SDK/Livox-SDK2))

After successful compilation please modify the path to the data in `run_sim.launch', then launch it.

The data can be downloaded here:

[rosbag](https://drive.google.com/file/d/1LrXRM73KUA1I1cU5NvOVeRSOPkXpkDKB/view?usp=drive_link)

[map](https://drive.google.com/file/d/19bfNp-ljfxNjLngdhIvIxfoClPygXRqC/view?usp=sharing)

# Testing on UWB-inertial fusion
Example for testing on [UTIL](https://utiasdsl.github.io/util-uwb-dataset/) (TDoA-inertial):
```
# Change bag_file and anchor_pose_path in `launch/run_util.launch` according to the path to the data
roslaunch ctgaupro run_util.launch
```
Evaluation using [evo](https://github.com/MichaelGrupp/evo) package
```
# Set if_save_traj in `launch/run_util.launch` to `1` and change traj_save_path accordingly
roslaunch ctgaupro run_util.launch
evo_ape tum /traj_save_path/gt.txt /traj_save_path/traj.txt -a --plot
```
Check the analytic Jacobian of IMU factor
```
roslaunch ctgaupro run_testui.launch
``` 
