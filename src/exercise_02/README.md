# Modeling and Control of Legged Robots - deliverable 2

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

## Launch rviz (for all exercises)
```sh
ros2 launch ros_visuals talos_rviz_world.launch.py
```

## Tutorial 4

### Exercise 1: Getting the robot to stand
```sh
ros2 run body_control standing
```

### Exercise 2: Getting the robot to balance on one foot
```sh
ros2 run body_control one_leg_stand
```

### Exercise 3 & 4: Getting the robot to do squats, Adding arm motions & Do some plotting
```sh
ros2 run body_control squatting
```
