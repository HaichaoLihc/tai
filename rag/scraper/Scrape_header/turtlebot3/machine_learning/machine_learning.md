
[Edit on GitHub](https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/machine_learning.md "https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/machine_learning.md") 

Kinetic 
Melodic
Noetic
Dashing
Foxy
Humble
Windows

# [Machine Learning](#machine-learning "#machine-learning")

Machine learning is a data analysis technique that teaches computers to recognize what is natural for people and animals - learning through experience. There are three types of machine learning: supervised learning, unsupervised learning, reinforcement learning.

This application is reinforcement learning with DQN (Deep Q-Learning). The reinforcement learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

This shows reinforcement learning with TurtleBot3 in gazebo.
This reinforcement learning is applied DQN(Deep Q-Learning) algorithm with LDS.  

We are preparing a four-step reinforcement learning tutorial.

Machine learning is a data analysis technique that teaches computers to recognize what is natural for people and animals - learning through experience. There are three types of machine learning: supervised learning, unsupervised learning, reinforcement learning.

This application is reinforcement learning with DQN (Deep Q-Learning). The reinforcement learning is concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

This shows reinforcement learning with TurtleBot3 in gazebo.
This reinforcement learning is applied DQN(Deep Q-Learning) algorithm with LDS.  

We are preparing a four-step reinforcement learning tutorial.

**NOTE**: This section is supported in ROS Kinetic and Melodic, and ROS2 Dashing.

Machine learning, learning through experience, is a data analysis technique that teaches computers to recognize what is natural for people and animals. There are three types of machine learning: supervised learning, unsupervised learning, reinforcement learning.

This application is reinforcement learning with DQN (Deep Q-Learning). The reinforcement learning is concerned with how software agents ought to take actions in an environment, so as to maximize some notion of cumulative reward.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

This shows reinforcement learning with TurtleBot3 in gazebo.
This reinforcement learning is applied DQN(Deep Q-Learning) algorithm with LDS.  

We are preparing a four-step reinforcement learning tutorial.

**NOTE**: This section is supported in ROS Kinetic and Melodic, and ROS2 Dashing.

**NOTE**: This section is supported in ROS Kinetic and Melodic, and ROS2 Dashing.

**NOTE**: This section is supported in ROS Kinetic and Melodic, and ROS2 Dashing.

## [Software Setup](#software-setup "#software-setup")

To do this tutorial, you need to install Tensorflow, Keras and Anaconda with Ubuntu 16.04 and ROS1 Kinetic.

### [Anaconda](#anaconda "#anaconda")

You can download [Anaconda 5.2.0](https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh "https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh") for Python 2.7 version.

After downloading Andaconda, go to the directory where the downloaded file is located at and enter the follow command.

Review and accept the license terms by entering `yes` in the terminal.  

Also add the Anaconda2 install location to PATH in the .basrhc file.

```
$ bash Anaconda2-5.2.0-Linux-x86_64.sh

```

After installing Anaconda,

```
$ source ~/.bashrc
$ python -V

```

If Anaconda is installed successfuly, `Python 2.7.xx :: Anaconda, Inc.` will be returned in the terminal.

### [ROS dependency packages](#ros-dependency-packages "#ros-dependency-packages")

Install required packages first.

```
$ pip install msgpack argparse

```

To use ROS and Anaconda together, you must additionally install ROS dependency packages.

```
$ pip install -U rosinstall empy defusedxml netifaces

```

### [Tensorflow](#tensorflow "#tensorflow")

This tutorial uses python 2.7(CPU only). If you want to use another python version and GPU, please refer to [TensorFlow](https://www.tensorflow.org/install/ "https://www.tensorflow.org/install/").

```
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl

```

### [Keras](#keras "#keras")

[Keras](https://keras.io/ "https://keras.io/") is a high-level neural networks API, written in Python and capable of running on top of TensorFlow.

```
$ pip install keras==2.1.5

```

Incompatible error messages regarding the tensorboard can be ignored as it is not used in this example, but it can be resolved by installing tensorboard as below.

```
$ pip install tensorboard

```

### [Machine Learning packages](#machine-learning-packages "#machine-learning-packages")

**WARNING**: Please install [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3 "https://github.com/ROBOTIS-GIT/turtlebot3"), [turtlebot3\_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs "https://github.com/ROBOTIS-GIT/turtlebot3_msgs") and [turtlebot3\_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations "https://github.com/ROBOTIS-GIT/turtlebot3_simulations") package before installing this package.

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
$ cd ~/catkin_ws && catkin_make

```

Machine Learning is running on a Gazebo simulation world. If you haven’t installed the TurtleBot3 simulation package, please install with the command below.

```
$ cd ~/catkin_ws/src/
$ git clone -b kinetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ~/catkin_ws && catkin_make

```

Completely uninstall and reinstall numpy to rectify problems. You may need to perform uninstall a few times until numpy is completely uninstalled.

```
$ pip uninstall numpy
$ pip show numpy
$ pip uninstall numpy
$ pip show numpy

```

At this point, numpy should be completed uninstalled and you should not see any numpy information when entering `pip show numpy`.

Reinstall the numpy.

```
$ pip install numpy pyqtgraph

```

To do this tutorial, you need to install Tensorflow, Keras and Anaconda with Ubuntu 18.04 and ROS1 Melodic.

### [Anaconda](#anaconda "#anaconda")

You can download [Anaconda 5.2.0](https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh "https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh") for Python 2.7 version.

After downloading Andaconda, go to the directory where the downloaded file is located at and enter the follow command.

Review and accept the license terms by entering `yes` in the terminal.  

Also add the Anaconda2 install location to PATH in the .basrhc file.

```
$ bash Anaconda2-5.2.0-Linux-x86_64.sh

```

After installing Anaconda,

```
$ source ~/.bashrc
$ python -V

```

If Anaconda is installed successfuly, `Python 2.7.xx :: Anaconda, Inc.` will be returned in the terminal.

### [ROS dependency packages](#ros-dependency-packages "#ros-dependency-packages")

Install required packages first.

```
$ pip install msgpack argparse

```

To use ROS and Anaconda together, you must additionally install ROS dependency packages.

```
$ pip install -U rosinstall empy defusedxml netifaces

```

### [Tensorflow](#tensorflow "#tensorflow")

This tutorial uses python 2.7(CPU only). If you want to use another python version and GPU, please refer to [TensorFlow](https://www.tensorflow.org/install/ "https://www.tensorflow.org/install/").

```
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl

```

### [Keras](#keras "#keras")

[Keras](https://keras.io/ "https://keras.io/") is a high-level neural networks API, written in Python and capable of running on top of TensorFlow.

```
$ pip install keras==2.1.5

```

Incompatible error messages regarding the tensorboard can be ignored as it is not used in this example, but it can be resolved by installing tensorboard as below.

```
$ pip install tensorboard

```

### [Machine Learning packages](#machine-learning-packages "#machine-learning-packages")

**WARNING**: Please install [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3 "https://github.com/ROBOTIS-GIT/turtlebot3"), [turtlebot3\_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs "https://github.com/ROBOTIS-GIT/turtlebot3_msgs") and [turtlebot3\_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations "https://github.com/ROBOTIS-GIT/turtlebot3_simulations") package before installing this package.

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
$ cd ~/catkin_ws && catkin_make

```

Machine Learning is running on a Gazebo simulation world. If you haven’t installed the TurtleBot3 simulation package, please install with the command below.

```
$ cd ~/catkin_ws/src/
$ git clone -b melodic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ~/catkin_ws && catkin_make

```

Completely uninstall and reinstall numpy to rectify problems. You may need to perform uninstall a few times until numpy is completely uninstalled.

```
$ pip uninstall numpy
$ pip show numpy
$ pip uninstall numpy
$ pip show numpy

```

At this point, numpy should be completed uninstalled and you should not see any numpy information when entering `pip show numpy`.

Reinstall the numpy.

```
$ pip install numpy pyqtgraph

```

Install Tensorflow and Keras on PC (Requirement: Ubuntu 18.04 and ROS2 Dashing)

### [ROS dependency packages](#ros-dependency-packages "#ros-dependency-packages")

1. Open a terminal.
2. Install ROS dependency packages for Python 3.6 version.

```
$ pip3 install -U rosinstall msgpack empy defusedxml netifaces

```

### [Tensorflow](#tensorflow "#tensorflow")

This instruction describes how to install TensorFlow in terminal.

1. Open a terminal.
2. Install TensorFlow.

```
$ pip3 install -U tensorflow

```

In the tutorial, python 3.6(CPU only) is used. To use different python version and GPU, refer to [TensorFlow](https://www.tensorflow.org/install/ "https://www.tensorflow.org/install/") installation page.

```
$ pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl

```

### [Keras](#keras "#keras")

[Keras](https://keras.io/ "https://keras.io/") is a high-level neural networks API, written in Python and capable of running on top of TensorFlow.

1. Open a terminal
2. Install Keras.

```
$ pip3 install keras==2.1.5

```

### [Machine Learning packages](#machine-learning-packages "#machine-learning-packages")

**WARNING**: Be sure to install [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3 "https://github.com/ROBOTIS-GIT/turtlebot3"), [turtlebot3\_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs "https://github.com/ROBOTIS-GIT/turtlebot3_msgs") and [turtlebot3\_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations "https://github.com/ROBOTIS-GIT/turtlebot3_simulations") package before installation of machine learning packages.

1. Open a terminal.
2. Install turtlebot3\_machine\_learning packages.

```
$ cd ~/robotis_ws/src/
$ git clone -b ros2 https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning.git
$ cd ~/robotis_ws && colcon build --symlink-install

```

## [Set parameters](#set-parameters "#set-parameters")

The goal of DQN Agent is to get the TurtleBot3 to the goal avoiding obstacles. When TurtleBot3 gets closer to the goal, it gets a positive reward, and when it gets farther it gets a negative reward.
The episode ends when the TurtleBot3 crashes on an obstacle or after a certain period of time. During the episode, TurtleBot3 gets a big positive reward when it gets to the goal, and TurtleBot3 gets a big negative reward when it crashes on an obstacle.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

### [Set state](#set-state "#set-state")

State is an observation of environment and describes the current situation. Here, `state_size` is 26 and has 24 LDS values, distance to goal, and angle to goal.

Turtlebot3’s LDS default is set to 360. You can modify sample of LDS at `turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro`.

```
<xacro:arg name="laser\_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`

```

```
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>

```

|  |  |
| --- | --- |
| **sample = 360** | **sample = 24** |

### [Set action](#set-action "#set-action")

Action is what an agent can do in each state. Here, turtlebot3 has always 0.15 m/s of linear velocity. angular velocity is determined by action.

| Action | Angular velocity(rad/s) |
| --- | --- |
| 0 | -1.5 |
| 1 | -0.75 |
| 2 | 0 |
| 3 | 0.75 |
| 4 | 1.5 |

### [Set reward](#set-reward "#set-reward")

When turtlebot3 takes an action in a state, it receives a reward. The reward design is very important for learning. A reward can be positive or negative. When turtlebot3 gets to the goal, it gets big positive reward. When turtlebot3
collides with an obstacle, it gets big negative reward. If you want to apply your reward design, modify `setReward` function at `/turtlebot3_machine_learning/turtlebot3_dqn/src/turtlebot3_dqn/environment_stage_#.py`.

### [Set hyper parameters](#set-hyper-parameters "#set-hyper-parameters")

This tutorial has been learned using DQN. DQN is a reinforcement learning method that selects a deep neural network by approximating the action-value function(Q-value). Agent has follow hyper parameters at `/turtlebot3_machine_learning/turtlebot3_dqn/nodes/turtlebot3_dqn_stage_#`.

| Hyper parameter | default | description |
| --- | --- | --- |
| episode\_step | 6000 | The time step of one episode. |
| target\_update | 2000 | Update rate of target network. |
| discount\_factor | 0.99 | Represents how much future events lose their value according to how far away. |
| learning\_rate | 0.00025 | Learning speed. If the value is too large, learning does not work well, and if it is too small, learning time is long. |
| epsilon | 1.0 | The probability of choosing a random action. |
| epsilon\_decay | 0.99 | Reduction rate of epsilon. When one episode ends, the epsilon reduce. |
| epsilon\_min | 0.05 | The minimum of epsilon. |
| batch\_size | 64 | Size of a group of training samples. |
| train\_start | 64 | Start training if the replay memory size is greater than 64. |
| memory | 1000000 | The size of replay memory. |

The goal of DQN Agent is to get the TurtleBot3 to the goal avoiding obstacles. When TurtleBot3 gets closer to the goal, it gets a positive reward, and when it gets farther it gets a negative reward.
The episode ends when the TurtleBot3 crashes on an obstacle or after a certain period of time. During the episode, TurtleBot3 gets a big positive reward when it gets to the goal, and TurtleBot3 gets a big negative reward when it crashes on an obstacle.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

### [Set state](#set-state "#set-state")

State is an observation of environment and describes the current situation. Here, `state_size` is 26 and has 24 LDS values, distance to goal, and angle to goal.

Turtlebot3’s LDS default is set to 360. You can modify sample of LDS at `turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro`.

```
<xacro:arg name="laser\_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`

```

```
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>

```

|  |  |
| --- | --- |
| **sample = 360** | **sample = 24** |

### [Set action](#set-action "#set-action")

Action is what an agent can do in each state. Here, turtlebot3 has always 0.15 m/s of linear velocity. angular velocity is determined by action.

| Action | Angular velocity(rad/s) |
| --- | --- |
| 0 | -1.5 |
| 1 | -0.75 |
| 2 | 0 |
| 3 | 0.75 |
| 4 | 1.5 |

### [Set reward](#set-reward "#set-reward")

When turtlebot3 takes an action in a state, it receives a reward. The reward design is very important for learning. A reward can be positive or negative. When turtlebot3 gets to the goal, it gets big positive reward. When turtlebot3
collides with an obstacle, it gets big negative reward. If you want to apply your reward design, modify `setReward` function at `/turtlebot3_machine_learning/turtlebot3_dqn/src/turtlebot3_dqn/environment_stage_#.py`.

### [Set hyper parameters](#set-hyper-parameters "#set-hyper-parameters")

This tutorial has been learned using DQN. DQN is a reinforcement learning method that selects a deep neural network by approximating the action-value function(Q-value). Agent has follow hyper parameters at `/turtlebot3_machine_learning/turtlebot3_dqn/nodes/turtlebot3_dqn_stage_#`.

| Hyper parameter | default | description |
| --- | --- | --- |
| episode\_step | 6000 | The time step of one episode. |
| target\_update | 2000 | Update rate of target network. |
| discount\_factor | 0.99 | Represents how much future events lose their value according to how far away. |
| learning\_rate | 0.00025 | Learning speed. If the value is too large, learning does not work well, and if it is too small, learning time is long. |
| epsilon | 1.0 | The probability of choosing a random action. |
| epsilon\_decay | 0.99 | Reduction rate of epsilon. When one episode ends, the epsilon reduce. |
| epsilon\_min | 0.05 | The minimum of epsilon. |
| batch\_size | 64 | Size of a group of training samples. |
| train\_start | 64 | Start training if the replay memory size is greater than 64. |
| memory | 1000000 | The size of replay memory. |

The goal of DQN Agent is to get TurtleBot3 to the goal to avoid obstacles. 
The closer TurtleBot3 gets to, the more positive reward it gets. When TurtleBot3 gets closer to the goal, it gets a positive reward. When it gets farther it gets a negative reward.
The episode ends when the TurtleBot3 crashes on an obstacle or after a certain period of time. During the episode, TurtleBot3 gets a big positive reward when it gets to the goal, and TurtleBot3 gets a big negative reward when it crashes on an obstacle.

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

### [Set State](#set-state "#set-state")

State is an observation of environment and describes the current situation. Here, `state_size` is 26 and has 24 LDS values, distance to goal, and angle to goal.

Turtlebot3’s LDS default is set to 360. You can modify sample of LDS at `turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro`.

```
<xacro:arg name="laser\_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`

```

```
<scan>
  <horizontal>
    <samples>360</samples>            # The number of sample. Modify it to 24
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>

```

|  |  |
| --- | --- |
| **sample = 360** | **sample = 24** |

### [Set Action](#set-action "#set-action")

Action is what an agent can do in each state. Here, turtlebot3 has always 0.15 m/s of linear velocity. angular velocity is determined by action.

| Action | Angular velocity(rad/s) |
| --- | --- |
| 0 | -1.5 |
| 1 | -0.75 |
| 2 | 0 |
| 3 | 0.75 |
| 4 | 1.5 |

### [Set Reward](#set-reward "#set-reward")

When turtlebot3 takes an action in a state, it receives a reward. The reward design is very important for learning. A reward can be positive or negative. When turtlebot3 gets to the goal, it gets big positive reward. When turtlebot3
collides with an obstacle, it gets big negative reward. If you want to apply your reward design, modify `setReward` function at `/turtlebot3_machine_learning/turtlebot3_dqn/src/turtlebot3_dqn/environment_stage_#.py`.

### [Set Hyper Parameters](#set-hyper-parameters "#set-hyper-parameters")

This tutorial has been learned using DQN. DQN is a reinforcement learning method that selects a deep neural network by approximating the action-value function(Q-value). Agent has follow hyper parameters at `/turtlebot3_machine_learning/turtlebot3_dqn/nodes/turtlebot3_dqn_stage_#`.

| Hyper parameter | default | description |
| --- | --- | --- |
| episode\_step | 6000 | The time step of one episode. |
| target\_update | 2000 | Update rate of target network. |
| discount\_factor | 0.99 | Represents how much future events lose their value according to how far away. |
| learning\_rate | 0.00025 | Learning speed. If the value is too large, learning does not work well, and if it is too small, learning time is long. |
| epsilon | 1.0 | The probability of choosing a random action. |
| epsilon\_decay | 0.99 | Reduction rate of epsilon. When one episode ends, the epsilon reduce. |
| epsilon\_min | 0.05 | The minimum of epsilon. |
| batch\_size | 64 | Size of a group of training samples. |
| train\_start | 64 | Start training if the replay memory size is greater than 64. |
| memory | 1000000 | The size of replay memory. |

## [Run Machine Learning](#run-machine-learning "#run-machine-learning")

In this Machine Learning example, 24 Lidar samples are used, which should be modified as written in the [Set parameters](#set-parameters "#set-parameters") section.

### [Stage 1 (No Obstacle)](#stage-1-no-obstacle "#stage-1-no-obstacle")

Stage 1 is a 4x4 map with no obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_1.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_1.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 2 (Static Obstacle)](#stage-2-static-obstacle "#stage-2-static-obstacle")

Stage 2 is a 4x4 map with four cylinders of static obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_2.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_2.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 3 (Moving Obstacle)](#stage-3-moving-obstacle "#stage-3-moving-obstacle")

Stage 2 is a 4x4 map with four cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_3.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_3.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_3.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 4 (Combination Obstacle)](#stage-4-combination-obstacle "#stage-4-combination-obstacle")

Stage 4 is a 5x5 map with walls and two cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_4.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_4.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

In this Machine Learning example, 24 Lidar samples are used, which should be modified as written in the [Set parameters](#set-parameters "#set-parameters") section.

### [Stage 1 (No Obstacle)](#stage-1-no-obstacle "#stage-1-no-obstacle")

Stage 1 is a 4x4 map with no obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_1.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_1.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_1.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 2 (Static Obstacle)](#stage-2-static-obstacle "#stage-2-static-obstacle")

Stage 2 is a 4x4 map with four cylinders of static obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_2.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_2.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_2.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 3 (Moving Obstacle)](#stage-3-moving-obstacle "#stage-3-moving-obstacle")

Stage 2 is a 4x4 map with four cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_3.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_3.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_3.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

### [Stage 4 (Combination Obstacle)](#stage-4-combination-obstacle "#stage-4-combination-obstacle")

Stage 4 is a 5x5 map with walls and two cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_4.jpg)

```
$ roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch

```

Open another terminal and enter the command below.

```
$ roslaunch turtlebot3_dqn turtlebot3_dqn_stage_4.launch

```

If you want to see the visualized data, launch the graph.

```
$ roslaunch turtlebot3_dqn result_graph.launch

```

The contents in e-Manual are subject to be updated without a prior notice. Therefore, some video may differ from the contents in e-Manual.

### [Stage 1 (No Obstacle)](#stage-1-no-obstacle "#stage-1-no-obstacle")

Stage 1 is a 4x4 map with no obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_1.jpg)

1. Open a terminal.
2. Bring the stage 1 in Gazebo map.

```
$ ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage1.launch.py
$ ros2 run turtlebot3_dqn dqn_gazebo 1
$ ros2 run turtlebot3_dqn dqn_environment
$ ros2 run turtlebot3_dqn dqn_agent 1

```

* If you want to test your trained model, use the following command.

```
$ ros2 run turtlebot3_dqn dqn_test 1

```

### [Stage 2 (Static Obstacle)](#stage-2-static-obstacle "#stage-2-static-obstacle")

Stage 2 is a 4x4 map with four cylinders of static obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_2.jpg)

1. Open a terminal.
2. Bring the stage 2 in Gazebo map.

```
$ ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage2.launch.py
$ ros2 run turtlebot3_dqn dqn_gazebo 2
$ ros2 run turtlebot3_dqn dqn_environment
$ ros2 run turtlebot3_dqn dqn_agent 2

```

* If you want to test your trained model, use the following command.

```
$ ros2 run turtlebot3_dqn dqn_test 2

```

### [Stage 3 (Moving Obstacle)](#stage-3-moving-obstacle "#stage-3-moving-obstacle")

Stage 3 is a 4x4 map with four cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_3.jpg)

1. Open a terminal.
2. Bring the stage 3 in Gazebo map.

```
$ ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage3.launch.py
$ ros2 run turtlebot3_dqn dqn_gazebo 3
$ ros2 run turtlebot3_dqn dqn_environment
$ ros2 run turtlebot3_dqn dqn_agent 3

```

* If you want to test your trained model, use the following command.

```
$ ros2 run turtlebot3_dqn dqn_test 3

```

### [Stage 4 (Combination Obstacle)](#stage-4-combination-obstacle "#stage-4-combination-obstacle")

Stage 4 is a 5x5 map with walls and two cylinders of moving obstacles.

![](/assets/images/platform/turtlebot3/machine_learning/stage_4.jpg)

1. Open a terminal.
2. Bring the stage 4 in Gazebo map.

```
$ ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py
$ ros2 run turtlebot3_dqn dqn_gazebo 4
$ ros2 run turtlebot3_dqn dqn_environment
$ ros2 run turtlebot3_dqn dqn_agent 4

```

* If you want to test your trained model, use the following command.

```
$ ros2 run turtlebot3_dqn dqn_test 4

```

 Previous Page
Next Page 