# Modeling and Control of Legged Robots - deliverable 1

## Build with
```sh
colcon build --symlink-install
```

## Tutorial 1

### Launch rviz (for each exercise)
```sh
ros2 launch ros_visuals launch.py
```

### exercise 1: SE(3)
```sh
ros2 run ros_visuals t11
```

### exercise 2: Twist
```sh
ros2 run ros_visuals t12
```

### exercise 3: Wrench
```sh
ros2 run ros_visuals t13
```

## Tutorial 2

### exercise 1: Create a simulation
```sh
ros2 run bullet_sims t2_temp
```

### exercise 2: Joint space controller
```sh
ros2 run bullet_sims t21
```

### exercise 3: Home posture controller
```sh
ros2 run bullet_sims t22
```

### exercise 4: Robot state visualization
Start rviz visualization
```sh
ros2 launch ros_visuals talos_rviz.launch.py
```
Start simulation
```sh
ros2 run bullet_sims t23
```


## Tutorial 3
Start rviz visualization
```sh
ros2 launch ros_visuals talos_rviz.launch.py
```
Start simulation
```sh
ros2 run bullet_sims t3_main
```
Start teleoperation
```sh
ros2 run bullet_sims teleoperation
```
