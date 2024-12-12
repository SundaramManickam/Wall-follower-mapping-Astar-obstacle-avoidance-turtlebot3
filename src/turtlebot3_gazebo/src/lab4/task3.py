#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
import numpy as np
from copy import deepcopy,copy
import itertools
from sklearn.linear_model import LinearRegression
from scipy import optimize
from scipy.spatial import ConvexHull
from sensor_msgs.msg import LaserScan, Image
from visualization_msgs.msg import MarkerArray, Marker
import warnings
import cv2
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd
import math
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist, Point
import PIL.Image
from PIL import  ImageOps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from graphviz import Graph
from nav_msgs.msg import Path
from std_msgs.msg import Header
import heapq

warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        im = PIL.Image.open(map_name)
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
        self.dist[sn.name] = 0
        heapq.heappush(self.sets, (self.__get_f_score(sn.name), sn.name)) 
        # Process until the queue is empty
        while self.sets:
            _, u = heapq.heappop(self.sets)
            # Check if the end node is reached
            if u == en.name:
                break
            current_node = self.in_tree.g[u]  
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
    
    
class Line:
    def __init__(self):
        self.slope: float = 0.0
        self.intercept: float = 0.0
        self.endpoints = [Point(), Point()]

    @property
    def length(self):
        return distance_between_points(self.endpoints[0], self.endpoints[1])


class Circle:
    def __init__(self):
        self.radius: float = float('inf')
        self.center = Point()


class Group:
    def __init__(self):
        self.points = []  # points that belong to a group
        self.best_fit_line = Line()
        self.best_fit_circle = Circle()

    @property
    def num_points(self):
        return len(self.points)

    def orthogonal_projection(self, pt):
        f_slope = 1.0 / (1.0 + (self.best_fit_line.slope * self.best_fit_line.slope))
        projected_point = Point()
        projected_point.x = float(f_slope * (pt.x + (self.best_fit_line.slope * pt.y) - (self.best_fit_line.slope * self.best_fit_line.intercept)))
        projected_point.y = float(f_slope * ((self.best_fit_line.slope * pt.x) + (self.best_fit_line.slope * self.best_fit_line.slope * pt.y) + self.best_fit_line.intercept))
        projected_point.z = 0.0
        return projected_point

    def get_longest_line(self):
        convex_hull_vertex_indices = ConvexHull(np.array([[pt.x, pt.y] for pt in self.points])).vertices
        convex_hull_vertices = [self.points[i] for i in convex_hull_vertex_indices]
        pairwise_points = list(itertools.combinations(convex_hull_vertices, 2))
        longest_length = 0
        end_point_1, end_point_2 = Point(), Point()
        for pair in pairwise_points:
            point1, point2 = pair[0], pair[1]
            d = distance_between_points(point1, point2)
            if d > longest_length:
                longest_length = d
                end_point_1.x, end_point_1.y = point1.x, point1.y
                end_point_2.x, end_point_2.y = point2.x, point2.y

        return end_point_1, end_point_2

    def calculate_best_fit_line(self):
        # create numpy 'vectors' of X and Y coordinates
        x_coordinates = np.array([pt.x for pt in self.points]).reshape((-1, 1))
        y_coordinates = np.array([pt.y for pt in self.points])

        # compute best fit line
        regressor = LinearRegression().fit(x_coordinates, y_coordinates)
        self.best_fit_line.slope = regressor.coef_
        self.best_fit_line.intercept = regressor.intercept_

        # project endpoints of longest line on the best fit line
        end_point_1, end_point_2 = self.get_longest_line()
        self.best_fit_line.endpoints[0] = self.orthogonal_projection(end_point_1)
        self.best_fit_line.endpoints[1] = self.orthogonal_projection(end_point_2)


    def _circle_fitting_least_squares_method(self):
        """
        Uses least square optimization
        (code from https://gist.github.com/lorenzoriano/6799568)
        """

        # if group has only 1 point, circle fitting fails (optimize.leastsq throws TypeError), so do nothing
        if len(self.points) <= 1:
            # should a warning be shown?
            return

        all_X_coordinates, all_Y_coordinates = [], []
        for point in self.points:
            all_X_coordinates.append(point.x)
            all_Y_coordinates.append(point.y)
        all_X_coordinates = np.array(all_X_coordinates)
        all_Y_coordinates = np.array(all_Y_coordinates)
        x_m = np.mean(all_X_coordinates)
        y_m = np.mean(all_Y_coordinates)

        def calc_R(x, y, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

        def f(c, x, y):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(x, y, *c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(f, center_estimate, args=(all_X_coordinates, all_Y_coordinates))
        self.best_fit_circle.center.x, self.best_fit_circle.center.y = center
        self.best_fit_circle.radius = calc_R(all_X_coordinates, all_Y_coordinates, *center).mean()

def distance_from_origin(p: Point):
    return np.sqrt(p.x ** 2 + p.y ** 2)


def distance_between_points(p1: Point, p2: Point):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def distance_point_from_line(end_point_a: Point, end_point_b: Point, any_point: Point):
    numerator = np.abs((end_point_a.x - end_point_b.x)*(end_point_b.y - any_point.y) -
                       (end_point_b.x - any_point.x)*(end_point_a.y - end_point_b.y))
    denominator = np.sqrt((end_point_a.x - end_point_b.x) ** 2 + (end_point_a.y - end_point_b.y) ** 2)
    return numerator / denominator

class Task3(Node):
    def __init__(self):
        super().__init__('task3')

        # declare parameters on Parameter Server and get their values
        self.declare_parameters(namespace='',
                                parameters=[('p_max_group_distance', 0.1),
                                            ('p_distance_proportion', 0.00628),
                                            ('p_max_split_distance', 0.2),
                                            ('p_min_group_points', 5),
                                            ('p_max_merge_separation', 0.02),
                                            ('p_max_merge_spread', 0.01),
                                            ('p_max_circle_radius', 0.5),
                                            ('p_radius_enlargement', 0.2),
                                            ('p_min_obstacle_size', 0.1)])
        self.p_max_group_distance = self.get_parameter('p_max_group_distance').value
        self.p_distance_proportion = self.get_parameter('p_distance_proportion').value
        self.p_max_split_distance = self.get_parameter('p_max_split_distance').value
        self.p_min_group_points = self.get_parameter('p_min_group_points').value
        self.p_max_merge_separation = self.get_parameter('p_max_merge_separation').value
        self.p_max_merge_spread = self.get_parameter('p_max_merge_spread').value
        self.p_max_circle_radius = self.get_parameter('p_max_circle_radius').value
        self.p_radius_enlargement = self.get_parameter('p_radius_enlargement').value
        self.p_min_obstacle_size = self.get_parameter('p_min_obstacle_size').value

        self.header = None
        self.img_pub = self.create_publisher(Image, 'video_data', 10)      

        # Initialize some empty lists
        self.points = []  # cartesian points (XY coordinates)
        self.groups = []  # list of Group() objects
        self._is_obstacle_detected_lines = []  # list of obstacles represented as lines
        self._is_obstacle_detected_circles = []  # list of obstacles represented as circles

        # for keeping track of which markers to delete
        self._all_marker_IDs = []

        # Subscribe to /scan topic on which input lidar data is available
        self.create_subscription(LaserScan, 'scan', self.callback_lidar_data,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.marker_pub = self.create_publisher(MarkerArray, 'marker', 10)
        self._is_obstacle_detected = False
        
        #NAVIGATION
        self.goal_pose = Pose()
        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Publishers
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.rate = self.create_rate(10)
        self.mp = MapProcessor('map')
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
        self.plan_done = False
        self.bridge = CvBridge()
        self.last_detected = False
    def reset_state(self):

        self.groups = []  # list of Group() objects
        self._is_obstacle_detected_lines = []  # list of obstacles represented as lines
        self._is_obstacle_detected_circles = []  # list of obstacles represented as circles

    def detect_obstacles(self):


        start_time = time.time()

        self.grouping()
        self.splitting()
        self.line_fitting()   # calculates best fit line
        self.segment_merging()
        self.circle_fitting()   # calculates best fit circle
        self.obstacle_classification()  # categorizes into line vs. circle obstacle
        self.remove_small_obstacles()
        

        # visualizing in RViz
        self.marker_pub.publish(self.visualize_groups_and_obstacles())


    def grouping(self):
       
        _time = time.time()

        points_remaining = deepcopy(self.points)

        while len(points_remaining) > 0:  # as points will be assigned to a group, they will be removed
            # start off a group with a point
            a_group = Group()
            point_from = points_remaining.pop(0)  # can be any point from points_remaining
            a_group.points.append(point_from)

            if len(points_remaining) == 0:  # in case that was the last point
                self.groups.append(a_group)
                break

            for p in points_remaining:
                r = distance_from_origin(p)
                d_from_prev_point = distance_between_points(p, point_from)

                if d_from_prev_point < self.p_max_group_distance + r * self.p_distance_proportion:  # add to group
                    a_group.points.append(p)
                    point_from = p

            self.groups.append(a_group)
            points_remaining = [point for point in points_remaining if point not in a_group.points]

        #                                                                                         (time.time() - _time)))

    def splitting(self):
        
        _time = time.time()

        groups_after_splitting = []

        for grp in self.groups:
            if grp.num_points <= self.p_min_group_points:
                continue

            end_point_1, end_point_2 = grp.get_longest_line()
            d_max = 0
            farthest_point = Point()
            for pt in grp.points:
                if pt in [end_point_1, end_point_2]:
                    continue  # end points of the longest line should not be used

                d_point_from_line = distance_point_from_line(end_point_1, end_point_2, pt)
                if d_point_from_line > d_max:
                    d_max = d_point_from_line
                    farthest_point.x = pt.x
                    farthest_point.y = pt.y

            r = distance_from_origin(farthest_point)
            splitting_threshold = self.p_max_split_distance + r * self.p_distance_proportion
            if d_max > splitting_threshold:
                grp1, grp2 = Group(), Group()
                grp1.points.append(end_point_1)
                grp2.points.append(end_point_2)

                grp1.points.append(farthest_point)
                grp2.points.append(farthest_point)

                for pt in grp.points:
                    if pt in [end_point_1, end_point_2, farthest_point]:
                        continue

                    dist_point_from_grp1_line = distance_point_from_line(end_point_1, farthest_point, pt)
                    dist_point_from_grp2_line = distance_point_from_line(farthest_point, end_point_2, pt)

                    if dist_point_from_grp1_line < dist_point_from_grp2_line:
                        grp1.points.append(pt)
                    else:
                        grp2.points.append(pt)

                groups_after_splitting.append(grp1)
                groups_after_splitting.append(grp2)

            else: 
                groups_after_splitting.append(grp)

        self.groups = groups_after_splitting
        #                                                                                          (time.time() - _time)))

    def line_fitting(self):
        """
        Regress a line to each group
        """
        _time = time.time()

        for grp in self.groups:
            grp.calculate_best_fit_line()


    def segment_merging(self):
        """

        Merge segments by checking 2 conditions
        """
        _time = time.time()

        merged_groups = []
        pairwise_groups = list(itertools.combinations(self.groups, 2))

        for every_pair in pairwise_groups:
            segments_close_enough = False
            group1, group2 = every_pair[0], every_pair[1]

            # get the 4 endpoints and check distances between them
            all_4_endpoints = [group1.best_fit_line.endpoints[0], group1.best_fit_line.endpoints[1],
                               group2.best_fit_line.endpoints[0], group2.best_fit_line.endpoints[1]]
            pairwise_points = list(itertools.combinations(all_4_endpoints, 2))
            for pair_of_pts in pairwise_points:
                pt1, pt2 = pair_of_pts[0], pair_of_pts[1]
                if distance_between_points(pt1, pt2) < self.p_max_merge_separation:
                    segments_close_enough = True
                    break

            if segments_close_enough:
                merged_grp = Group()
                merged_grp.points = group1.points
                merged_grp.points.extend(group2.points)

                # Spread test, to check if the lines are collinear
                merged_grp.calculate_best_fit_line()
                d_max = 0
                for pt in all_4_endpoints:
                    d = distance_point_from_line(merged_grp.best_fit_line.endpoints[0],
                                                 merged_grp.best_fit_line.endpoints[1],
                                                 pt)
                    if d > d_max:
                        d_max = d

                if d_max < self.p_max_merge_spread:  # confirmed that the 2 groups should be replace by one
                    merged_groups.append(merged_grp)

            else:
                merged_groups.append(group1)
                merged_groups.append(group2)

        self.groups = list(set(merged_groups))

    def circle_fitting(self):
        """
        Fits a circle
        """
        _time = time.time()

        # compute best fit circles for all the groups
        for grp in self.groups:
            grp._circle_fitting_least_squares_method()


    def obstacle_classification(self):
        """

        Categorizes obstacles as lines or circles.
        Only circles less than a certain radius is considered to be a circular obstacle.
        That's because some best fit lines can be very long and the circle for those obstacles would be too huge!
        """
        _time = time.time()

        # check whether circle (after enlargement) is to be added to list of circle obstacles or line obstacles
        for grp in self.groups:
            if grp.best_fit_circle.radius + self.p_radius_enlargement <= self.p_max_circle_radius:
                self._is_obstacle_detected_circles.append(grp)
                
            else:
                self._is_obstacle_detected_lines.append(grp)
    def remove_small_obstacles(self):
        """
        Only keep those obstacles which satisfy these conditions:
        - line obstacles with length >= p_min_obstacle_size
        - circle obstacles with length >= p_min_obstacle_size

        If p_min_obstacle_size is 0, then this method remove_small_obstacles() should have no effect.
        If p_min_obstacle_size is +infinity, then no obstacles, however big, will be remaining.
        """
        _time = time.time()

        # conditions in which to return preemptively
        if self.p_min_obstacle_size <= 0.0:
            return
        elif self.p_min_obstacle_size == float('inf'):
            self._is_obstacle_detected_lines = []
            self._is_obstacle_detected_circles = []
            return

        no_of_detected_lines, no_of_detected_circles = len(self._is_obstacle_detected_lines), len(self._is_obstacle_detected_circles)

        self._is_obstacle_detected_lines = [grp for grp in self._is_obstacle_detected_lines
                                if grp.best_fit_line.length >= self.p_min_obstacle_size]
        self._is_obstacle_detected_circles = [grp for grp in self._is_obstacle_detected_circles
                                  if grp.best_fit_circle.radius >= self.p_min_obstacle_size]

    def visualize_groups_and_obstacles(self):
        marker_list = []
        dummy_id = 0

      
        # Obstacles represented by circles
        for grp in self._is_obstacle_detected_circles:
            marker = Marker()
            marker.id = dummy_id
            marker.header = self.header
            marker.type = Marker.CYLINDER
            marker.action = 0  # 0 add/modify an object, 1 (deprecated), 2 deletes an object, 3 deletes all objects
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.scale.x, marker.scale.y = float(grp.best_fit_circle.radius * 2), float(
                grp.best_fit_circle.radius * 2)  # set different xy values for ellipse
            marker.scale.z = 0.1
            marker.pose.position.x = grp.best_fit_circle.center.x
            marker.pose.position.y = grp.best_fit_circle.center.y
            marker.pose.position.z = marker.scale.z / 2
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker_list.append(marker)
            if dummy_id not in self._all_marker_IDs:
                self._all_marker_IDs.append(dummy_id)
            dummy_id += 1

        # delete detections from previous message that are not present in current message
        for id_to_del in self._all_marker_IDs[dummy_id:]:
            marker = Marker()
            marker.id = id_to_del
            marker.action = Marker.DELETE
            marker_list.append(marker)
        self._all_marker_IDs = self._all_marker_IDs[:dummy_id]

        # publish
        marker_array = MarkerArray()
        marker_array.markers = marker_list
        return marker_array




    def image_callback(self,msg):
        
        try:
            image = self.bridge.imgmsg_to_cv2(msg)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            small_kernel = np.ones((5,5), np.uint8)
            #range of lower and upper
            mask = cv2.inRange(hsv,  np.array([0, 0, 0])  , np.array([255, 255, 15])  )
            check = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
            contours, _ = cv2.findContours(check, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 380000: 
                    self._is_obstacle_detected = True
                    self.move_ttbot(0.0,0.0)
                    self.get_logger().info("detected")
                    
                else:  
                    self._is_obstacle_detected = False
        except CvBridgeError as e:
            self.get_logger().error(f"error {e}")

    def callback_lidar_data(self, laser_scan):
        self.header = laser_scan.header
        theta = np.arange(laser_scan.angle_min, laser_scan.angle_max, laser_scan.angle_increment)
        r = np.array(laser_scan.ranges)

        if len(theta) != len(r):
            if len(theta) < len(r):
                r = r[:len(theta)]  # truncate r

            else:
                theta = theta[:len(r)]  # truncate theta


        # convert points from polar coordinates to cartesian coordinates
        indices_of_valid_r = [i for i in range(len(r)) if laser_scan.range_min <= r[i] < laser_scan.range_max]
        r, theta = [r[i] for i in indices_of_valid_r], [theta[i] for i in indices_of_valid_r]
        self.points = [Point(x=r[i] * np.cos(theta[i]), y=r[i] * np.sin(theta[i]), z=0.0)  # 2D lidar doesn't have Z
                       for i in range(len(r))]
            
        self.reset_state()
        self.detect_obstacles()
        if(len(self._is_obstacle_detected_circles) and not self.plan_done):
            for grp in self._is_obstacle_detected_circles:
                print(grp.best_fit_circle.radius)
                if(grp.best_fit_circle.radius > 0.20 and grp.best_fit_circle.radius < 0.25):
                    x = grp.best_fit_circle.center.x
                    y = grp.best_fit_circle.center.y                    
                    if(x<-0.1 and x>-0.5 and y<0.7 and y>-0.7):
                        print("BEHINDDDDDDDDDDD")
                        # print(str(x)+"   "+str(y))
                        # self.obstacles = True
                        if(y<=0):
                            self.move_ttbot(0.2,0.0)
                            time.sleep(0.25)
                            self.move_ttbot(0.0, 0.0)
                            
                        else:
                            
                            self.move_ttbot(0.2,0.0)
                            time.sleep(0.2)
                            self.move_ttbot(0.0, 0.0)
                            
                    if(x>0 and x<0.7 and y<0.5 and y>-0.5):
                        print("FRONTTTTTTTTTT")
                        print(str(x)+"   "+str(y))
                        self.move_ttbot(-0.3, 0.0)
                        time.sleep(0.25)
                        self.move_ttbot(0.0, 0.0)

    
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
                pose.pose.position.y = pose.pose.position.y  -0.1

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
        kp = 7
        kd = 4
        ki = 0.001
        dt = 0.1
        self.lin_int_error += error * dt
        derivative = (error - self.lin_prev_error) / dt
        self.lin_prev_error = error
        linear_velocity = (kp * error) + (ki * self.lin_int_error) + (kd * derivative)
        if math.isinf(linear_velocity):
            linear_velocity = 0.0
        linear_velocity = min(max(linear_velocity, 0.0), 0.4) 
        return linear_velocity
    
    def angular_pid(self,error):
        kp = 7
        kd = 15
        ki = 0.001
        dt = 0.1
        self.ang_int_error += error * dt
        derivative = (error - self.ang_prev_error) / dt
        self.ang_prev_error = error
        ang_vel = (kp * error) + (ki * self.ang_int_error) + (kd * derivative)
        ang_vel = min(max(ang_vel, 0.0), 0.3)
        return ang_vel
    
    def goal_reached(self, current, target, off=0.20):
        dx = target.position.x - current.position.x
        dy = target.position.y - current.position.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        # print("goal_reached function" + str(distance))
        return distance < off
    
    def move_ttbot(self, speed, heading):

        cmd_velocity = Twist()
        if(self._is_obstacle_detected or (speed ==0 and heading==0)):
            cmd_velocity.linear.x = float(0)
            cmd_velocity.angular.z = float(0)
        else:
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
            return len(self.node_path)-1

    def path_navigator(self,current_goal,prev_goal):
        rclpy.spin_once(self, timeout_sec=0.1)
        self.distance_to_goal = math.sqrt((current_goal.pose.position.x - self.ttbot_data_pose.position.x) ** 2 + (current_goal.pose.position.y - self.ttbot_data_pose.position.y) ** 2)
        target_angle = math.atan2(current_goal.pose.position.y - self.ttbot_data_pose.position.y, current_goal.pose.position.x - self.ttbot_data_pose.position.x)
        current_angle = self.get_yaw(self.ttbot_data_pose)  
        target_angle = self.normalize_angle(target_angle)
        yaw_error = self.normalize_angle(target_angle - current_angle)
        lin_err = 0.45
        ang_err = 0.13
        self.is_goal_reached = False
        if(abs(yaw_error) > ang_err):
            # print("yaw_error"+str(yaw_error))
            self.speed = 0.07*self.distance_to_goal
            self.heading = self.angular_pid(abs(yaw_error)) if yaw_error > 0 else -self.angular_pid(abs(yaw_error))
        elif ((self.distance_to_goal > lin_err)): 
            # print("lin"+str(self.distance_to_goal))
            self.speed = self.linear_pid(self.distance_to_goal)
            self.heading = 0.07*yaw_error
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
            # print("runing")
            
            self.ttbot_pose_tuple = tuple(map(int, starting.split(',')))
            if(self.node_path is not None):
                while not self.goal_reached(self.ttbot_data_pose,self.goal_pose):
                    rclpy.spin_once(self, timeout_sec=0.1)
                    self.idx = self.get_path_idx()
                    print("waypoint no:" + str(self.idx))
                    current_goal = self.path.poses[self.idx]
                    # print(self.idx)
                    while(not self.is_goal_reached):
                        rclpy.spin_once(self, timeout_sec=0.1)
                        self.path_navigator(current_goal,self.path.poses[self.last_idx])
                        # print(self.speed)
                        # print(self.heading)
                        self.move_ttbot(self.speed,self.heading)
                    self.last_idx = self.idx
                    self.is_goal_reached = False
                self.plan_done = True  
                self.get_logger().info("Goal reached, stopping robot")
                self.move_ttbot(0.0, 0.0)



def main(args=None):                       
    rclpy.init(args=args)

    task3 = Task3()
    try:
        # rclpy.spin(task3)
        task3.run()
        # print("runing")
    except KeyboardInterrupt:
        pass
    finally:
        task3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()