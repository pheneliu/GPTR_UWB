# Presiquite

* Please catkin build [SFUISE](https://github.com/KIT-ISAS/SFUISE) in your workspace to have the cf_msg, which is required in gptr.
* Please install Ceres 2.1.0 to run the examples and tests.

# Testing the lidar pipeline:

## With synthetic data



After successful compilation please modify the path to the data in `run_sim.launch', then launch it.

The data can be downloaded here:

[rosbag](https://drive.google.com/file/d/1LrXRM73KUA1I1cU5NvOVeRSOPkXpkDKB/view?usp=drive_link)

[map](https://drive.google.com/file/d/19bfNp-ljfxNjLngdhIvIxfoClPygXRqC/view?usp=sharing)

# Testing on UWB-inertial fusion
Example for testing on [UTIL](https://utiasdsl.github.io/util-uwb-dataset/) (TDoA-inertial):
```
# Change bag_file and anchor_pose_path in `launch/run_util.launch` according to the path to the data
roslaunch gptr run_util.launch
```
Baseline approach for comparison is the ESKF presented by the UTIL paper
Evaluation using [evo](https://github.com/MichaelGrupp/evo) package
```
# Set if_save_traj in `launch/run_util.launch` to `1` and change traj_save_path accordingly
roslaunch gptr run_util.launch
evo_ape tum /traj_save_path/gt.txt /traj_save_path/traj.txt -a --plot
```
Check the analytic Jacobian of IMU factor
```
roslaunch gptr run_testui.launch
``` 
