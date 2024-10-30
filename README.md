# Preresiquite

* Please catkin build [SFUISE](https://github.com/KIT-ISAS/SFUISE) in your workspace to have the cf_msg, which is required in gptr.
* Please install Ceres 2.1.0 to run the examples and tests.
* Git clone and catkin build the repo.
  
Please raise an issue should you encounter any issue with the compilation of the package.

# Testing the lidar pipeline:

## With synthetic data

You can download and unzip the file `cloud_avia_mid_dynamic_extrinsics` from [here](https://drive.google.com/file/d/1Q5fTn5OvWd_I2RvVfiUKir90q5HshzQM/view?usp=sharing). It contains the pointclouds and the prior map for the experiment.

After that, modify the path to the data and prior map in `run_sim.launch` and launch it. You should see the following visualization from rviz.

<img src="docs/sim.gif" alt="synthetic_exp" width="600"/>

## With handheld setup

Similar to the synthetic dataset, please download the data and the prior map from [here](https://drive.google.com/file/d/1QId8X4LFxYdYewHSBXiDEAvpIFD8w-ei/view?usp=sharing).

Then specify the paths to the data and prior map in `gptr/launch/run_lio_cathhs_iot.launch` before roslaunch. You should see the following illustration.

<img src="docs/cathhs.gif" alt="cathhs_exp" width="600"/>

## Evaluation

Please use the scripts `analysis_cathhs.ipynb` and `analysis_sim.ipynb` to evaluate the result.

<br/>

# Testing on UWB-inertial fusion

Please download the [UTIL](https://utiasdsl.github.io/util-uwb-dataset/) (TDoA-inertial) dataset.

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
