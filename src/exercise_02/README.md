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

## Launch rviz (Same for all exercises)
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


## Tutorial 5
##### Notes: 
- Check if plotting library is installed: `pip3 list | grep PyQt5`
  Install PyQt5 (version 5.15.10) for visualization
- In the case of using the visualization: To stop the program, please close the plot/diagram window.
- Threading is used to speed up the simulation. You can disable it by setting `THREADING = False`.
- To reduce plotting computation, you can limit the maximum number of time steps by enabling `USE_MAXLEN = True` and setting `MAXLEN = 500`.


### Exercise 1 & 2 & 3: Prepare your simulation, Ground reference points & Balance control
```sh
ros2 run body_control t51
```

### Exercise 4: Position-controlled hardware interface
```sh
ros2 run body_control t52
```
