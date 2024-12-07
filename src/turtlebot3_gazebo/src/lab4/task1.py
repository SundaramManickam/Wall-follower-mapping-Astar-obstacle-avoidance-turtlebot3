#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import numpy as np
import time

class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        # self.amcl_pose_subscriber = self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     '/amcl_pose',
        #     self.amcl_pose_callback,
        #     10
        # )

        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        # Publisher
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        self.ang_int_error = 0
        self.ang_prev_error = 0
        self.ang_side_int_error = 0
        self.ang_side_prev_error = 0
        self.lin_int_error = 0
        self.lin_prev_error = 0
        # State Variables
        self.scan_data = None
        self.ttbot_data_pose = Pose()
        self.yaw = 0.0
        self.state = 'FINDING_WALL'  # States: FINDING_WALL, ALIGNING_WITH_WALL, FOLLOWING_WALL, CORRECTING_PATH
        self.is_right = False
        self.destination_yaw = 0.0
        # Control Parameters
        self.forward_speed = 0.2
        self.turn_speed = 0.1
        self.wall_stop_distance = 0.7  # Desired distance from the wall

        # Timer for Main Logic
        self.timer = self.create_timer(0.1, self.wall_following_logic)

    def scan_callback(self, msg):
        """Callback for /scan topic to process laser scan data."""
        self.scan_data = msg.ranges

    def odom_callback(self, msg):
        """Callback for /odom topic to extract orientation."""
        orientation_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
    def imu_callback(self,msg):
        self.get_yaw(msg.orientation)
    def amcl_pose_callback(self, msg):
        """Callback for /amcl_pose to get the robot's estimated position."""
        self.ttbot_data_pose = msg.pose.pose
        
    def linear_pid(self,error):
        kp = 3
        kd = 0.5
        ki = 0.05
        dt = 0.1
        
        self.lin_int_error += error * dt
        derivative = (error - self.lin_prev_error) / dt
        self.lin_prev_error = error
        linear_velocity = (kp * error) + (ki * self.lin_int_error) + (kd * derivative)
        if math.isinf(linear_velocity):
            linear_velocity = 0.0
        linear_velocity = max(min(linear_velocity, 0.2), -0.1)# Clamp velocity to [0.0, 0.15]
        return linear_velocity
    
    def alignment_pid(self,error):
        kp = 7
        kd = 1.5
        ki = 0.01
        dt = 0.1
   
        self.ang_int_error += error * dt
        derivative = (error - self.ang_prev_error) / dt
        self.ang_prev_error = error
        
        ang_vel = (kp * error) + (ki * self.ang_int_error) + (kd * derivative)
        ang_vel = min(max(ang_vel, 0.0), 0.5)
            
        return ang_vel
    def side_distance_pid(self,error):
        kp = 0.6
        kd = 0.2
        ki = 0.01
        dt = 0.1
   
        self.ang_side_int_error += error * dt
        derivative = (error - self.ang_side_prev_error) / dt
        self.ang_side_prev_error = error
        
        ang_vel = (kp * error) + (ki * self.ang_side_int_error) + (kd * derivative)
        ang_vel = min(max(ang_vel, 0.0), 0.25)
            
        return ang_vel
    
    def normalize_angle(self,angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def set_destination_yaw(self,delta_angle):
        """Set a new destination yaw relative to the current yaw."""
        self.destination_yaw = self.normalize_angle(self.yaw + delta_angle)
        # print("destination_yaw" + str(self.destination_yaw))

    def get_yaw(self, orientation):
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        roll, pitch, self.yaw = euler_from_quaternion(quaternion)
        # print(self.yaw)
    

    def weighted_median(self,data, weights):
        sorted_data = sorted(zip(data, weights), key=lambda x: x[0])
        cum_weights = np.cumsum([w for _, w in sorted_data])
        median_idx = np.searchsorted(cum_weights, cum_weights[-1] / 2)
        return sorted_data[median_idx][0]
    
    def move_forward(self):
        twist = Twist()
        print("moving forwards")
        twist.linear.x = 0.3
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)
        time.sleep(1)
        twist.linear.x = 0.0
        self.cmd_vel_publisher.publish(twist)
        
    def wall_following_logic(self):
        """Main logic for wall-following behavior."""
        
        
        if not self.scan_data:
            self.get_logger().warning("Scan data not available.")
            return
        # print(self.scan_data[0])
        # return
        twist = Twist()
        # front_distance = self.scan_data[0]
        # if(math.isinf(front_distance)):
        #     front_distance = 11
        # self.distance_error = front_distance - self.wall_stop_distance

        front_indices = list(range(350, 359)) + list(range(0, 10))  # Indices for -10° to 10°
        front_high_indices = list(range(0, 35) )+ list(range(315, 359))   # Indices for -25° to 25°
        
        valid_distances = [self.scan_data[i] if not math.isinf(self.scan_data[i]) else 11 for i in front_indices]
        valid_high_distances = [(self.scan_data[i], i) if not math.isinf(self.scan_data[i]) else (11, i)for i in front_high_indices]

        
        
        
        # Threshold for stopping the robot
        threshold_distance = 0.3  # Example threshold distance

        
        front_distance = min(valid_distances)
        min_front_distance, min_index = min(valid_high_distances, key=lambda x: x[0])


        
            

        # Compute the distance error
        self.distance_error = front_distance - self.wall_stop_distance
        

        if self.state == 'FINDING_WALL':
            # Move forward to find the nearest wall
            # for i in range(-10,0):
            #     front_distance[i] = self.scan_data[i+360]  # 0° corresponds to front
            
            if self.distance_error <=0.05:
                self.get_logger().warn("Wall ---- ROTATE.")
                self.state = 'ROTATE'
                self.set_destination_yaw(math.pi / 2)  # Turn 90° left

            else:
                yaw_error = self.destination_yaw - self.yaw
                twist.linear.x = self.linear_pid(self.distance_error)
                twist.angular.z = 0.0
                
        elif self.state == 'ROTATE':
            yaw_error = self.normalize_angle(self.destination_yaw - self.yaw)
            # yaw_error = 0
            
            right_avg = sum(self.scan_data[266:271]) / 5  # Average for left half]
            left_avg = sum(self.scan_data[271:276]) / 5  # Average for right half
            alignment_error = left_avg - right_avg  # Positive = closer to right wall
            if( math.isinf(right_avg) or math.isinf(left_avg)):
                # self.get_logger().warn("INF just .")
                alignment_error =0
            # Calculate offset error
            # print("this is allignment error while rotating" + str(alignment_error))
            # print("this is distance error while rotating" + str(self.distance_error))
            
            # print(yaw_error)
            # is_side_looking_at_wall = (right_avg - self.wall_stop_distance <0.05 and left_avg - self.wall_stop_distance <0.05)
            if (abs(yaw_error) + abs(alignment_error) <= 0.07 or (abs(alignment_error)<=0.07 and abs(yaw_error)<0.7 and not self.is_right)):  # Close enough to the target angle
                
                
                #     print("sleep")
                #     time.sleep(0.5)
                # if(self.destination_yaw>0):
                #     print("sleep")
                # #     self.move_forward()
                #     time.sleep(0.5)
                #     print("sleep over")
                self.get_logger().warn("ROTATE - ALIGNING_WITH_WALL")
                self.state = 'FOLLOWING_WALL'
                self.ang_int_error = 0
                self.ang_prev_error = 0
                self.ang_side_int_error = 0
                self.ang_side_prev_error = 0
                self.lin_int_error = 0
                self.lin_prev_error = 0
            else:
                # self.get_logger().warn("ROTATE")
                
                # twist.angular.z = self.alignment_pid(abs(yaw_error+0.6*(alignment_error))) if (yaw_error+alignment_error) > 0 else -self.alignment_pid(abs(yaw_error+0.6*(alignment_error)))
                twist.angular.z = self.alignment_pid(abs(yaw_error)) if (yaw_error) > 0 else -self.alignment_pid(abs(yaw_error))
                
                # twist.linear.x =self.linear_pid(max(min(self.distance_error, 0.35), -0.35))
                twist.linear.x =self.linear_pid(self.distance_error)
                if(twist.angular.z >0.35 or twist.angular.z <-0.35):
                    twist.linear.x= twist.linear.x*0.7
                # print(min_front_distance)
                # print(min_index)
                if(min_front_distance<threshold_distance):
                    self.get_logger().info(".............................................reversing")
                    # print(min_front_distance)
                    twist.linear.x = -0.05
                    twist.angular.z = 0.0
                    # if( min_index in range(0,35)):
                    #     self.get_logger().info(".............................................clockwise")
                    #     # twist.angular.z = 0.08
                    # else:
                    #     self.get_logger().info(".............................................anticlockwise")
                    #     # twist.angular.z = -0.08
                else:
                    if(self.is_right):
                        twist.linear.x = 0.2       

                   
        # elif self.state == 'ALIGNING_WITH_WALL':
        #     # Use side LIDAR data to align with the wall
            
        #     right_avg = sum(self.scan_data[266:271]) / 5  # Average for left half]
        #     left_avg = sum(self.scan_data[271:276]) / 5  # Average for right half
            
        #     # Calculate offset error
        #     alignment_error = left_avg - right_avg  # Positive = closer to right wall
        #     if( math.isinf(right_avg) or math.isinf(left_avg)):
        #         self.get_logger().warn("ALIGNING_WITH_WALL - FOLLOWING_WALL.")
        #         self.state='FOLLOWING_WALL'
        #         alignment_error =0
            

        #     if abs(alignment_error) <= 0.05:  # Threshold for alignment
        #         self.get_logger().warn("Aligned with wall. Switching to FOLLOWING_WALL.")
        #         self.state = 'FOLLOWING_WALL'
        #     else:
        #         # Rotate left or right based on alignment error
        #         twist = Twist()
        #         # self.get_logger().warn("ALIGHNING WITH WALL")
                
        #         twist.angular.z = self.alignment_pid(abs(alignment_error)) if alignment_error > 0 else -self.alignment_pid(abs(alignment_error))
        #         twist.linear.x = self.linear_pid(max(min(self.distance_error, 0.2), -0.2))  # Stop forward motion during alignment
        #         if(twist.angular.z >0.2 or twist.angular.z <-0.2):
        #             twist.linear.x= twist.linear.x*1

        elif self.state == 'FOLLOWING_WALL':
            # Maintain wall on the right side
            side_distance = self.scan_data[270]  # Check distance at 270°
            desired_distance = 0.45  # Desired distance from the wall (in meters)

            # Side distance error
            if not math.isinf(side_distance):  # Ensure LIDAR data is valid
                side_distance_error = side_distance - desired_distance 
            else:
                side_distance_error = 0

            self.is_right = False
            #allignment error#####
            right_avg = sum(self.scan_data[266:271]) / 5  # Average for left half]
            left_avg = sum(self.scan_data[271:276]) / 5  # Average for right half
                        # Calculate offset error
            alignment_error =  left_avg - right_avg   # Positive = closer to right wall
            # right_distance = avg(self.scan_data[266:276])
            # print(right_distance)
            if( math.isinf(right_avg) or math.isinf(left_avg)):
                # self.get_logger().warn("INF just.")
                alignment_error =  0  # Positive = closer to right wall


            # self.get_logger().warn(f"Alignment Error: {alignment_error:.2f}")
            # self.get_logger().warn(f"Side Distance Error: {side_distance_error:.2f}")
            # self.get_logger().warn(f"Left Avg: {left_avg:.2f}, Right Avg: {right_avg:.2f}")
            # self.get_logger().info(f"FRONT_DISTANCE: {front_distance:.2f}, SIDE_DISTANCE: {side_distance:.2f}")

            if (front_distance >0.2 and front_distance < self.wall_stop_distance):
                self.get_logger().warn("CLOSING")
                self.state="ROTATE"
                self.is_right = False
                self.set_destination_yaw(math.pi / 2)  # Turn 90° left
                self.ang_int_error = 0
                self.ang_prev_error = 0
                self.ang_side_int_error = 0
                self.ang_side_prev_error = 0
                self.lin_int_error = 0
                self.lin_prev_error = 0
            elif ((right_avg+left_avg)/2 > self.wall_stop_distance+0.5):  # Wall opening detected
                self.get_logger().warn("OPENING")
                self.state = 'ROTATE'
                self.is_right = True
                self.set_destination_yaw(-math.pi / 2)  # Turn 90° right
                self.ang_int_error = 0
                self.ang_prev_error = 0
                self.ang_side_int_error = 0
                self.ang_side_prev_error = 0
                self.lin_int_error = 0
                self.lin_prev_error = 0
            else:
                angular_correction = -self.alignment_pid(abs(alignment_error)) if alignment_error > 0 else self.alignment_pid(abs(alignment_error))
                side_correction = -self.side_distance_pid(abs(side_distance_error)) if side_distance_error > 0 else self.side_distance_pid(abs(side_distance_error))
                # print("SIDE_CORRECTION" + str(side_correction))
                # twist.angular.z = self.alignment_pid(abs(yaw_error+alignment_error)) if yaw_error+alignment_error > 0 else -self.alignment_pid(abs(yaw_error+alignment_error))
                twist.angular.z = (angular_correction + side_correction)
                # print(min_front_distance)
                # print(min_index)
                # print("min_front" + str(min_front_distance))
                if(min_front_distance<threshold_distance):
                    self.get_logger().info("...........................reversing")
                    # print(min_front_distance)
                    twist.linear.x = -0.05
                    # twist.angular.z = 0.0
                    # if( min_index in range(0,35)):
                    #     twist.angular.z = 0.08
                    # else:
                    #     twist.angular.z = -0.08
                    
                else:
                    twist.linear.x = self.linear_pid(self.distance_error)
                    if(twist.angular.z >0.2 or twist.angular.z <-0.2):
                        twist.linear.x= twist.linear.x/2
                    
        # print("angular speed:"+str(twist.angular.z))
        # print("linear speed:"+str(twist.linear.x))
                


        self.cmd_vel_publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
