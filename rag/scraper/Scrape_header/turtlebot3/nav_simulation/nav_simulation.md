
[Edit on GitHub](https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/simulation/nav_simulation.md "https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/simulation/nav_simulation.md") 

Kinetic 
Melodic
Noetic
Dashing
Foxy
Humble
Windows

## [Navigation Simulation](#navigation-simulation "#navigation-simulation")

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation node.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.   

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `2D Nav Goal` button in the RViz menu.   

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation node.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.   

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `2D Nav Goal` button in the RViz menu.   

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation node.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `2D Nav Goal` button in the RViz menu.   

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation2. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation2 node.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation2. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation2 node.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

Just like the SLAM in Gazebo simulator, you can select or create various environments and robot models in virtual Navigation world. However, proper map has to be prepared before running the Navigation2. Other than preparing simulation environment instead of bringing up the robot, Navigation Simulation is pretty similar to that of [Navigation](/docs/en/platform/turtlebot3/navigation/#navigation "/docs/en/platform/turtlebot3/navigation/#navigation").

### Launch Simulation World

Terminate all applications with `Ctrl` + `C` that were launced in the previous sections.

In the previous [SLAM](/docs/en/platform/turtlebot3/slam/#slam "/docs/en/platform/turtlebot3/slam/#slam") section, TurtleBot3 World is used to creat a map. The same Gazebo environment will be used for Navigation.

Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

```

![](/assets/images/icon_unfold.png) Read more about **How to load TurtleBot3 House**

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py

```

### Run Navigation Node

Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and run the Navigation2 node.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True map:=$HOME/map.yaml

```

### [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

### [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

**NOTE**: This feature is available for Kinetic, Noetic, Dashing, Foxy.

 Previous Page
Next Page 