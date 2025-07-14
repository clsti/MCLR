# Modeling and Control of Legged Robots - deliverable 3

## Instructions
* Create workspace
```sh
mkdir ros2_ws && cd ros2_ws
```
* Copy src folder here and build with
```sh
colcon build --symlink-install
```
* Source workspace
```sh
source install/setup.bash
```
* Execute all commands in workspace

## Command for rviz visualization of robot
```sh
ros2 launch ros_visuals talos_rviz_world.launch.py
```

## Tutorial 6

### Task 1: Getting familiar
```sh
python3 src/exercise_03/lmpc_walking/example_2_pydrake.py
```

### Task 2: Linear inverted pendulum
```sh
python3 src/exercise_03/lmpc_walking/ocp_lipm_2ord.py
```

### Task 3 Linear Model Predictive Control
```sh
python3 src/exercise_03/lmpc_walking/mpc_lipm_2ord.py
```


## Tutorial 7
Group: Timo Class, Victor Velezmoro

Unfortunately, we encountered some unexpected issues:
The robot's support foot is rotating after setting the foot contacts. This can lead to instabilities and the robot might crash. Even after removing all rotations and assuming constant rotation as described in the tutorial sheet, the problem still persists. Since simulation contacts can be quite noisy, as mentioned in the lecture, the robot's ability to walk, and due to time constraints and the Tutorial deadline, we were unable to investigate this issue further.
In the case that the robot is not walking properly, we have included a video and all plots for reference in the folder '''src/exercise_03/T7_images'''. 
We hope this is acceptable and would be very interested in any further insights or explanations regarding this issue.


### Task 1: Implement Modules

#### Foot Trajectory
```sh
python3 src/exercise_03/walking_control/walking_control/foot_trajectory.py
```

#### Foot Step planner
```sh
python3 src/exercise_03/walking_control/walking_control/footstep_planner.py
```

#### LIP MPC
```sh
ros2 run walking_control lip_mpc
```

#### Robot
```sh
ros2 run walking_control talos
```

### Task 2: Implement the Walking Cycle
```sh
ros2 run walking_control talos
```