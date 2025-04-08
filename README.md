This was a final project for the course ME597 Autonomous System. 
This code has three tasks
1. Task1 - Autonomously map any surrounding - This was achieved by using a wall following method from the LIDAR sensor along with PID tuning to make sure to always maintain distance and control velocity.
2. Task2 - A star global path planner from Point A to Point B
3. Task3 - A static obstacle avoidance method - The detection part was achieved using two methods, 1. Lidar to detect circle of particular size (size of trash can which is obstacle) and 2. using computer vision to detect thrash can's colour. The avoidance is done by eitehr pausing at position or speeding straight through if its already in line of movement of trash can.   

# ME597 sim_ws
## Instructions:
1. Simply save this workspace e.g., 
    ```
    cd ~/ros2 # cd into the dir you want to keep this workspace in
    mkdir src
    cd src
    git clone https://github.com/SundaramManickam/Wall-follower-mapping-Astar-obstacle-avoidance-turtlebot3.git
    ```

2. In a new terminal build the sim_ws workspace: 
    ```
    cd sim_ws
    colcon build --symlink-install
    ```

3. Add turtlebot3 environment variable to .bashrc file
    ```
    echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
    ```
4. Run these to install useful packages you will use in the simulator.
    ```
    sudo apt install ros-humble-turtlebot3-teleop
    sudo apt install ros-humble-slam-toolbox
    sudo apt install ros-humble-navigation2
    ```
    
    ```
    pip install pynput
    ```

5. Don't forget to source the workspace whenever you use it
    ```
    cd sim_ws
    source install/local_setup.bash
    ```

## Demonstration
### Task1
![Task 1](videos/task_1.gif)
### Task2
![Task 2](videos/task_2.gif)
### Task3
![Task 3](videos/task3.gif)
### Bonus Task 
![Bonus](videos/bonus.gif)
