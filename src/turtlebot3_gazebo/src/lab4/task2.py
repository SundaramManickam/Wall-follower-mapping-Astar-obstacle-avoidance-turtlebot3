#!/usr/bin/env python3


import pandas as pd
from copy import copy
import time
import math
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphviz import Graph
from tf_transformations import euler_from_quaternion
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Header
import heapq


class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits,self.old_limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):

        map_name = 'src/turtlebot3_gazebo/maps/' + map_name
        f = open( map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        map_name = map_df.image[0]
        map_name = 'src/turtlebot3_gazebo/maps/' + map_name
        im = Image.open(map_name)
        im_width,im_height = im.size
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]
        xmin_old = map_df.origin[0][0]
        xmax_old = map_df.origin[0][0] + im_width * map_df.resolution[0]
        ymin_old = map_df.origin[0][1]
        ymax_old = map_df.origin[0][1] + im_height * map_df.resolution[0]
        print(im.size)
        print(xmin,xmax,ymin,ymax)
        print(xmin_old,xmax_old,ymin_old,ymax_old)

        return im, map_df, [xmin,xmax,ymin,ymax],[xmin_old,xmax_old,ymin_old,ymax_old]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255
        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array

class Noding():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)

class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Noding('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path_array

class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True

class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.sets = []  # Priority queue for the open set
        self.dist = {name: np.inf for name, node in in_tree.g.items()}
        self.h = {name: 0 for name, node in in_tree.g.items()}
        # Initialize heuristic values based on Euclidean distance
        for name, node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

        # Via dictionary to keep track of the path
        self.via = {name: 0 for name, node in in_tree.g.items()}

    def __get_f_score(self, node):
        # Calculate f-score as the sum of g (distance from start) and h (heuristic)
        return self.dist[node] + self.h[node]

    def solve(self, sn, en):
        # Set the start node distance to 0
        self.dist[sn.name] = 0
        heapq.heappush(self.sets, (self.__get_f_score(sn.name), sn.name))  # Use node name instead of the object
        # Process until the queue is empty
        while self.sets:
            _, u = heapq.heappop(self.sets)
            # Check if the end node is reached
            if u == en.name:
                break
            current_node = self.in_tree.g[u]  # Get the actual node object
            # Iterate through each child of the current node
            for i in range(len(current_node.children)):
                c = current_node.children[i]
                w = current_node.weight[i]
                new_dist = self.dist[current_node.name] + w
                # Update distance and path if a shorter path is found
                if new_dist < self.dist[c.name]:
                    self.dist[c.name] = new_dist
                    self.via[c.name] = current_node.name
                     # Push to the open set with updated f-score
                    heapq.heappush(self.sets, (self.__get_f_score(c.name), c.name))  # Use node name


    def reconstruct_path(self, sn, en):
        start_key = sn.name
        end_key = en.name
        dist = self.dist[end_key]
        u = end_key
        path = []
        while u != start_key:
            path.append(u)
            u = self.via[u]
        path.reverse()
        return path, dist




class Task2(Node):
    """! Task2 node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Task2'):
        """! Class constructor.
        @param  None.
        @return An instance of the Task2 class.
        """
        super().__init__(node_name)
        self.goal_pose = Pose()
        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.rate = self.create_rate(10)
        self.mp = MapProcessor('sync_classroom_map')
        kr = self.mp.rect_kernel(11,11)
        self.mp.inflate_map(kr,True)
        self.mp.get_graph_from_map()
        self.ttbot_data_pose = Pose()
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.path = Path()
        self.path.header = Header()
        self.path.header.frame_id = "map" 
        self.idx = 1
        self.angle = 0
        self.distance_to_goal = 100
        self.speed = 0.0
        self.heading = 0.0
        self.lin_int_error = 0
        self.lin_prev_error = 0
        self.ang_int_error = 0
        self.ang_prev_error = 0
        self.last_idx = 0
        self.node_path = None
        self.is_goal_reached = False
    
    
    def __real_world_to_grid(self, data):        
        
        #scale found from grid (200,141) and (occupancy grid corners)
        xscale = self.mp.map.map_im.size[0]/(self.mp.map.old_limits[1]-self.mp.map.old_limits[0])
        yscale = self.mp.map.map_im.size[1]/(self.mp.map.old_limits[3]-self.mp.map.old_limits[2])

        #offset by the origin from the left bottom corner 
        pixel_y = (data.position.x+(-1*self.mp.map.old_limits[0]))*xscale
        pixel_x = (self.mp.map.old_limits[3] - data.position.y)*yscale 

        return str(int(pixel_x)) +","+ str(int(pixel_y))  

    def __grid_to_real_world(self, pixel_x,pixel_y):
        
        # xscale =200/14.85
        # yscale = 141/10.5
        #scale found from grid (200,141) and (occupancy grid corners)
        xscale = self.mp.map.map_im.size[0]/(self.mp.map.old_limits[1]-self.mp.map.old_limits[0])
        yscale = self.mp.map.map_im.size[1]/(self.mp.map.old_limits[3]-self.mp.map.old_limits[2])
        
        #offset by the origin from the left bottom corner
        world_x = (pixel_y/xscale)-(-1*self.mp.map.old_limits[0])
        world_y = self.mp.map.old_limits[3] - (pixel_x/yscale)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"  
        pose_stamped.pose.position.x = world_x
        pose_stamped.pose.position.y = world_y
        pose_stamped.pose.position.z = 0.0  
        pose_stamped.pose.orientation.w = 1.0 

        return pose_stamped
    
    
    def __goal_pose_cbk(self, data):
        self.goal_pose = Pose()
        self.goal_pose= data.pose
        if(self.ttbot_data_pose is not None):
            path,self.node_path = self.a_star_path_planner(self.ttbot_data_pose, self.goal_pose)
            # Apply scaling and offset to each pose in the Path message
            for pose in self.path.poses:
                pose.pose.position.x = pose.pose.position.x  
                pose.pose.position.y = pose.pose.position.y  -0.06

            self.path_pub.publish(self.path)
        else:
            print("try again no current pose")

    def __ttbot_pose_cbk(self, data):

        msg = Pose()
        msg.position.x = float(math.ceil(data.pose.pose.position.x))
        msg.position.y = float(math.ceil(data.pose.pose.position.y))
        self.ttbot_data_pose = data.pose.pose
        
    

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        self.path = Path()
        self.path.header = Header()
        self.path.header.frame_id = "map" 
        self.last_idx = 0
        self.idx = 1
        ending = self.__real_world_to_grid(end_pose) #ttbot_pose in string
        starting = self.__real_world_to_grid(start_pose) #ttbot_pose in string
        self.ttbot_pose_tuple = tuple(map(int, starting.split(',')))
        self.get_logger().info(
                'A* planner.\n> start: {},\n> end: {}'.format(starting, ending))
        if(ending !="" and starting != ""):
            self.mp.map_graph.root = starting
            self.mp.map_graph.end = ending
            as_maze = AStar(self.mp.map_graph)
            start = time.time()
            as_maze.solve(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
            end = time.time()
            self.get_logger().info('Elapsed Time:{:.4f}'.format(end - start))
            node_path,dist_as = as_maze.reconstruct_path(self.mp.map_graph.g[self.mp.map_graph.root],self.mp.map_graph.g[self.mp.map_graph.end])
            self.get_logger().info(
                'A* planner.\n>  {}'.format(node_path)   )
            path_arr_as = self.mp.draw_path(node_path)
            for coordinate in node_path:
                x,y = map(int, coordinate.split(','))
                pose= self.__grid_to_real_world(x,y)
                self.path.poses.append(pose)

            return self.path, node_path
        else:
            self.get_logger().info(
                'No new plan'   )
            node_path = 0
    
    def get_yaw(self, pose):
        orientation = pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        return yaw  
    
    def linear_pid(self,error):
        kp = 4
        kd = 1.5
        ki = 0.0
        dt = 0.1
        self.lin_int_error += error * dt
        derivative = (error - self.lin_prev_error) / dt
        self.lin_prev_error = error
        linear_velocity = (kp * error) + (ki * self.lin_int_error) + (kd * derivative)
        if math.isinf(linear_velocity):
            linear_velocity = 0.0
        linear_velocity = min(max(linear_velocity, 0.0), 0.2)  # Clamp velocity to [0.0, 0.15]
        return linear_velocity
    
    def angular_pid(self,error):
        kp = 3
        kd = 1.5
        ki = 0.01
        dt = 0.1
        self.ang_int_error += error * dt
        derivative = (error - self.ang_prev_error) / dt
        self.ang_prev_error = error
        ang_vel = (kp * error) + (ki * self.ang_int_error) + (kd * derivative)
        ang_vel = min(max(ang_vel, 0.0), 0.2)
        return ang_vel
    
    def goal_reached(self, current, target, off=0.30):
        dx = target.position.x - current.position.x
        dy = target.position.y - current.position.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        print("goal_reached function" + str(distance))
        return distance < off
    
    def move_ttbot(self, speed, heading):

        cmd_velocity = Twist()
        cmd_velocity.linear.x = float(speed)
        cmd_velocity.angular.z = float(heading)
        self.cmd_publisher.publish(cmd_velocity)

    def normalize_angle(self,angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_path_idx(self):
        if(self.last_idx!=len(self.node_path)-1):
            return self.last_idx+1
        else:
            # print("fdsdsds")
            # print(self.last_idx)
            # print(len(self.node_path)-1)            
            return len(self.node_path)-1 

    def path_navigator(self,current_goal,prev_goal):
        rclpy.spin_once(self, timeout_sec=0.1)
        self.distance_to_goal = math.sqrt((current_goal.pose.position.x - self.ttbot_data_pose.position.x) ** 2 + (current_goal.pose.position.y - self.ttbot_data_pose.position.y) ** 2)
        target_angle = math.atan2(current_goal.pose.position.y - self.ttbot_data_pose.position.y, current_goal.pose.position.x - self.ttbot_data_pose.position.x)
        current_angle = self.get_yaw(self.ttbot_data_pose)  
        target_angle = self.normalize_angle(target_angle)
        yaw_error = self.normalize_angle(target_angle - current_angle)
        lin_err = 0.33
        ang_err = 0.12
        self.is_goal_reached = False
        if(abs(yaw_error) > ang_err):
            
            self.speed = 0.005
            self.heading = self.angular_pid(abs(yaw_error)) if yaw_error > 0 else -self.angular_pid(abs(yaw_error))
        elif ((self.distance_to_goal > lin_err)): 
            self.speed = self.linear_pid(self.distance_to_goal)
            self.heading = 0
        else: 
            self.is_goal_reached = True
            self.speed = 0
            self.heading = 0 
            self.lin_int_error = 0
            self.ang_int_error = 0
            self.lin_prev_error =0
            self.ang_prev_error = 0

        
    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        """
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            starting = self.__real_world_to_grid(self.ttbot_data_pose)
            self.ttbot_pose_tuple = tuple(map(int, starting.split(',')))
            if(self.node_path is not None):
                while not self.goal_reached(self.ttbot_data_pose,self.goal_pose):
                    rclpy.spin_once(self, timeout_sec=0.1)
                    self.idx = self.get_path_idx()
                    print("waypoint no:" + str(self.idx))
                    current_goal = self.path.poses[self.idx]
                    while(not self.is_goal_reached):
                        rclpy.spin_once(self, timeout_sec=0.1)
                        self.path_navigator(current_goal,self.path.poses[self.last_idx])
                        self.move_ttbot(self.speed,self.heading)
                    self.last_idx = self.idx
                    self.is_goal_reached = False
                    
                self.get_logger().info("Goal reached, stopping robot")
                self.move_ttbot(0.0, 0.0)

def main(args=None):
    
    rclpy.init(args=args)
    task2 = Task2(node_name='Task2')

    try:
        task2.run()
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
