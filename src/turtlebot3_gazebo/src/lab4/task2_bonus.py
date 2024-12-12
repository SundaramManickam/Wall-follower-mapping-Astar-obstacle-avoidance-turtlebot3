

import cv2
import numpy as np
import math
import random
import argparse
import os

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []
        self.cost = float('inf')  # Initialize the cost to infinity

def rewire(node_list, new_node, radius,img):
    """Rewire the tree to ensure optimal connections"""
    for i in range(len(node_list)):
        dist, _ = calculate_distance_angle(node_list[i].x, node_list[i].y, new_node.x, new_node.y)
        if dist < radius and not detect_collision(new_node.x, new_node.y, node_list[i].x, node_list[i].y,img):
            new_cost = new_node.cost + dist
            if new_cost < node_list[i].cost:
                node_list[i].cost = new_cost
                node_list[i].parent_x = new_node.parent_x.copy()
                node_list[i].parent_y = new_node.parent_y.copy()

def detect_collision(x_start, y_start, x_end, y_end, image):
    pixel_colors = []
    x_coords = np.linspace(x_start, x_end, num=100)
    y_coords = np.linspace(y_start, y_end, num=100)

    for index in range(len(x_coords)):
        x_pix, y_pix = int(round(x_coords[index])), int(round(y_coords[index]))
        if 0 <= x_pix < image.shape[1] and 0 <= y_pix < image.shape[0]:
            current_pixel = image[y_pix, x_pix]
            # Check for black (occupied) or gray (unknown)
            if current_pixel == 0 or (10 <= current_pixel <= 220):  # Adjust thresholds for gray
                return True  # Collision detected
    return False  # No collision

# Evaluate collision with obstacle and adjust

def evaluate_collision(x_initial, y_initial, x_target, y_target, image):
    _, angle = calculate_distance_angle(x_target, y_target, x_initial, y_initial)
    next_x = x_target + stepSize * np.cos(angle)
    next_y = y_target + stepSize * np.sin(angle)
    print(x_target, y_target, x_initial, y_initial)
    print("Angle:", angle)
    print("Next Point:", next_x, next_y)

    # Ensure the point stays within image boundaries
    height, width = image.shape
    if next_y < 0 or next_y > height or next_x < 0 or next_x > width:
        print("Point is out of bounds")
        direct_connection = False
        node_connection = False
    else:
        # Evaluate direct connection to endpoint
        if detect_collision(next_x, next_y, end[0], end[1], image):
            direct_connection = False
        else:
            direct_connection = True

        # Evaluate connection between intermediate nodes
        if detect_collision(next_x, next_y, x_target, y_target, image):
            node_connection = False
        else:
            node_connection = True

    return (next_x, next_y, direct_connection, node_connection)

# Calculate distance and angle between two points
def calculate_distance_angle(x_a, y_a, x_b, y_b):
    distance = math.sqrt(((x_a - x_b) ** 2) + ((y_a - y_b) ** 2))
    angle = math.atan2(y_b - y_a, x_b - x_a)
    return (distance, angle)

# Identify the nearest node index
def find_nearest_node(x_pos, y_pos):
    distances = []
    for node in node_list:
        distance, _ = calculate_distance_angle(x_pos, y_pos, node.x, node.y)
        distances.append(distance)
    return distances.index(min(distances))

# Generate a random point within image bounds
def generate_random_point(height, width):
    rand_y = random.randint(0, height)
    rand_x = random.randint(0, width)
    return (rand_x, rand_y)

# Rapidly-exploring Random Tree algorithm
def RRT(image, image_copy, start_point, goal_point, step_size):
    height, width = image.shape
    node_list[0] = Nodes(start_point[0], start_point[1])
    node_list[0].parent_x.append(start_point[0])
    node_list[0].parent_y.append(start_point[1])

    # Display start and goal points
    cv2.circle(image_copy, (start_point[0], start_point[1]), 5, (0, 0, 255), thickness=3, lineType=8)
    cv2.circle(image_copy, (goal_point[0], goal_point[1]), 5, (0, 0, 255), thickness=3, lineType=8)

    i = 1
    path_found = False
    while not path_found:
        rand_x, rand_y = generate_random_point(height, width)
        print("Random points:", rand_x, rand_y)

        nearest_index = find_nearest_node(rand_x, rand_y)
        nearest_x = node_list[nearest_index].x
        nearest_y = node_list[nearest_index].y
        print("Nearest node coordinates:", nearest_x, nearest_y)

        # Check direct connection
        new_x, new_y, direct_connection, node_connection = evaluate_collision(rand_x, rand_y, nearest_x, nearest_y, image)
        print("Check collision:", new_x, new_y, direct_connection, node_connection)

        if direct_connection and node_connection:
            print("Node can connect directly with goal")
            node_list.append(i)
            node_list[i] = Nodes(new_x, new_y)
            node_list[i].parent_x = node_list[nearest_index].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_index].parent_y.copy()
            node_list[i].parent_x.append(new_x)
            node_list[i].parent_y.append(new_y)

            cv2.circle(image_copy, (int(new_x), int(new_y)), 2, (0, 0, 255), thickness=3, lineType=8)
            cv2.line(image_copy, (int(new_x), int(new_y)), (int(node_list[nearest_index].x), int(node_list[nearest_index].y)), (0, 255, 0), thickness=1, lineType=8)
            cv2.line(image_copy, (int(new_x), int(new_y)), (goal_point[0], goal_point[1]), (255, 0, 0), thickness=2, lineType=8)

            print("Path has been found")
            for j in range(len(node_list[i].parent_x) - 1):
                cv2.line(image_copy, (int(node_list[i].parent_x[j]), int(node_list[i].parent_y[j])), (int(node_list[i].parent_x[j + 1]), int(node_list[i].parent_y[j + 1])), (255, 0, 0), thickness=2, lineType=8)
            # cv2.imwrite("media/" + str(i) + ".jpg", image_copy)
            cv2.imwrite("rrt.jpg", image_copy)
            break

        elif node_connection:
            print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(new_x, new_y)
            node_list[i].parent_x = node_list[nearest_index].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_index].parent_y.copy()
            node_list[i].parent_x.append(new_x)
            node_list[i].parent_y.append(new_y)
            i += 1

            # Display
            cv2.circle(image_copy, (int(new_x), int(new_y)), 2, (0, 0, 255), thickness=3, lineType=8)
            cv2.line(image_copy, (int(new_x), int(new_y)), (int(node_list[nearest_index].x), int(node_list[nearest_index].y)), (0, 255, 0), thickness=1, lineType=8)
            # cv2.imwrite("media/" + str(i) + ".jpg", image_copy)
            cv2.imshow("sdc", image_copy)
            cv2.waitKey(1)
            continue

        else:
            print("No direct connection and no node connection. Generating new random points.")
            continue

def RRT_star(image, image_display, start_point, end_point, step_size, search_radius):
    height, width = image.shape
    node_list[0] = Nodes(start_point[0], start_point[1])
    node_list[0].cost = 0  # Starting node has zero cost
    node_list[0].parent_x.append(start_point[0])
    node_list[0].parent_y.append(start_point[1])

    cv2.circle(image_display, (start_point[0], start_point[1]), 5, (0, 0, 255), thickness=3, lineType=8)
    cv2.circle(image_display, (end_point[0], end_point[1]), 5, (0, 0, 255), thickness=3, lineType=8)

    index = 1
    path_found = False
    while not path_found:
        random_x, random_y = generate_random_point(height, width)
        nearest_index = find_nearest_node(random_x, random_y)
        nearest_node_x = node_list[nearest_index].x
        nearest_node_y = node_list[nearest_index].y

        next_x, next_y, direct_connection, node_connection = evaluate_collision(random_x, random_y, nearest_node_x, nearest_node_y, image)
        if node_connection:
            new_node = Nodes(next_x, next_y)
            new_node.parent_x = node_list[nearest_index].parent_x.copy()
            new_node.parent_y = node_list[nearest_index].parent_y.copy()
            new_node.parent_x.append(next_x)
            new_node.parent_y.append(next_y)
            new_node.cost = node_list[nearest_index].cost + step_size
            node_list.append(new_node)

            cv2.circle(image_display, (int(next_x), int(next_y)), 2, (0, 0, 255), thickness=3, lineType=8)
            cv2.line(image_display, (int(next_x), int(next_y)), (int(nearest_node_x), int(nearest_node_y)), (0, 255, 0), thickness=1, lineType=8)

            # Rewire
            rewire(node_list, new_node, search_radius, image)

            # Check direct connection to the endpoint
            if not detect_collision(next_x, next_y, end_point[0], end_point[1], image):
                path_found = True
                cv2.line(image_display, (int(next_x), int(next_y)), (end_point[0], end_point[1]), (255, 0, 0), thickness=2, lineType=8)
                final_node = Nodes(end_point[0], end_point[1])
                final_node.parent_x = new_node.parent_x.copy()
                final_node.parent_y = new_node.parent_y.copy()
                final_node.parent_x.append(end_point[0])
                final_node.parent_y.append(end_point[1])
                node_list.append(final_node)

                for j in range(len(final_node.parent_x) - 1):
                    cv2.line(image_display, 
                             (int(final_node.parent_x[j]), int(final_node.parent_y[j])),
                             (int(final_node.parent_x[j + 1]), int(final_node.parent_y[j + 1])),
                             (255, 0, 0), thickness=2, lineType=8)

                cv2.imwrite("rrt_star.jpg", image_display)
                break

    return
    
def draw_circle(event,x,y,flags,param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2,(x,y),5,(255,0,0),-1)
        coordinates.append(x)
        coordinates.append(y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='world2.png',metavar='ImagePath', action='store', dest='imagePath',
                    help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=10,metavar='Stepsize', action='store', dest='stepSize',
                    help='Step-size to be used for RRT branches')
    parser.add_argument('-start', type=int, default=[20,20], metavar='startCoord', dest='start', nargs='+',
                    help='Starting position in the maze')
    parser.add_argument('-stop', type=int, default=[450,250], metavar='stopCoord', dest='stop', nargs='+',
                    help='End position in the maze')
    parser.add_argument('-selectPoint', help='Select start and end points from figure', action='store_true')

    args = parser.parse_args()

    # remove previously stored data
    try:
      os.system("rm -rf media")
    except:
      print("Dir already clean")
    os.mkdir("media")

    img = cv2.imread(args.imagePath,0) # load grayscale maze image
    img2 = cv2.imread(args.imagePath) # load colored maze image
    start = tuple(args.start) #(20,20) # starting coordinate
    end = tuple(args.stop) #(450,250) # target coordinate
    stepSize = args.stepSize # stepsize for RRT
    node_list = [0] # list to store all the node points

    coordinates=[]
    if args.selectPoint:
        print("Select start and end points by double clicking, press 'escape' to exit")
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)
        while(1):
            cv2.imshow('image',img2)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        # print(coordinates)
        start=(coordinates[0],coordinates[1])
        end=(coordinates[2],coordinates[3])

    # run the RRT algorithm 
    RRT(img, img2, start, end, stepSize)
    radius = stepSize * 5  # Define radius for rewiring
    # RRT_star(img, img2, start, end, stepSize, radius)