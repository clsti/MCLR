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