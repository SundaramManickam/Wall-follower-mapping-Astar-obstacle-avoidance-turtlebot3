"""

Path planning with Rapidly-Exploring Random Trees (RRT)

author: Aakash(@nimrobotics)
web: nimrobotics.github.io

"""

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
        dist, _ = dist_and_angle(node_list[i].x, node_list[i].y, new_node.x, new_node.y)
        if dist < radius and not collision(new_node.x, new_node.y, node_list[i].x, node_list[i].y,img):
            new_cost = new_node.cost + dist
            if new_cost < node_list[i].cost:
                node_list[i].cost = new_cost
                node_list[i].parent_x = new_node.parent_x.copy()
                node_list[i].parent_y = new_node.parent_y.copy()
                # node_list[i].parent_x.append(node_list[i].x)
                # node_list[i].parent_y.append(node_list[i].y)

def collision(x1, y1, x2, y2, img):
    color = []
    x = np.linspace(x1, x2, num=100)
    y = np.linspace(y1, y2, num=100)

    for i in range(len(x)):
        ix, iy = int(round(x[i])), int(round(y[i]))
        if 0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]:
            pixel_value = img[iy, ix]
            # Check for black (occupied) or gray (unknown)
            if pixel_value == 0 or (10 <= pixel_value <= 220):  # Adjust thresholds for gray
                return True  # Collision detected
    return False  # No collision


# check the  collision with obstacle and trim
def check_collision(x1,y1,x2,y2,img):
    _,theta = dist_and_angle(x2,y2,x1,y1)
    x=x2 + stepSize*np.cos(theta)
    y=y2 + stepSize*np.sin(theta)
    print(x2,y2,x1,y1)
    print("theta",theta)
    print("check_collision",x,y)

    # TODO: trim the branch if its going out of image area
    # print("Image shape",img.shape)
    hy,hx=img.shape
    if y<0 or y>hy or x<0 or x>hx:
        print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(x,y,end[0],end[1],img):
            directCon = False
        else:
            directCon=True

        # check connection between two nodes
        if collision(x,y,x2,y2,img):
            nodeCon = False
        else:
            nodeCon = True

    return(x,y,directCon,nodeCon)

# return dist and angle b/w new point and nearest node
def dist_and_angle(x1,y1,x2,y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2-y1, x2-x1)
    return(dist,angle)

# return the neaerst node index
def nearest_node(x,y):
    temp_dist=[]
    for i in range(len(node_list)):
        dist,_ = dist_and_angle(x,y,node_list[i].x,node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))

# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x,new_y)

def RRT_star(img, img2, start, end, stepSize, radius):
    h, l = img.shape
    node_list[0] = Nodes(start[0], start[1])
    node_list[0].cost = 0  # Starting node has zero cost
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    cv2.circle(img2, (start[0], start[1]), 5, (0, 0, 255), thickness=3, lineType=8)
    cv2.circle(img2, (end[0], end[1]), 5, (0, 0, 255), thickness=3, lineType=8)

    i = 1
    pathFound = False
    while not pathFound:
        nx, ny = rnd_point(h, l)
        nearest_ind = nearest_node(nx, ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        tx, ty, directCon, nodeCon = check_collision(nx, ny, nearest_x, nearest_y,img)
        if nodeCon:
            new_node = Nodes(tx, ty)
            new_node.parent_x = node_list[nearest_ind].parent_x.copy()
            new_node.parent_y = node_list[nearest_ind].parent_y.copy()
            new_node.parent_x.append(tx)
            new_node.parent_y.append(ty)
            new_node.cost = node_list[nearest_ind].cost + stepSize
            node_list.append(new_node)

            cv2.circle(img2, (int(tx), int(ty)), 2, (0, 0, 255), thickness=3, lineType=8)
            cv2.line(img2, (int(tx), int(ty)), (int(nearest_x), int(nearest_y)), (0, 255, 0), thickness=1, lineType=8)

            # Rewire
            rewire(node_list, new_node, radius,img)

            # Check direct connection to the end
            if not collision(tx, ty, end[0], end[1],img):
                pathFound = True
                cv2.line(img2, (int(tx), int(ty)), (end[0], end[1]), (255, 0, 0), thickness=2, lineType=8)
                final_node = Nodes(end[0], end[1])
                final_node.parent_x = new_node.parent_x.copy()
                final_node.parent_y = new_node.parent_y.copy()
                final_node.parent_x.append(end[0])
                final_node.parent_y.append(end[1])
                node_list.append(final_node)

                for j in range(len(final_node.parent_x) - 1):
                    cv2.line(img2, 
                             (int(final_node.parent_x[j]), int(final_node.parent_y[j])),
                             (int(final_node.parent_x[j + 1]), int(final_node.parent_y[j + 1])),
                             (255, 0, 0), thickness=2, lineType=8)

                cv2.imwrite("out_star.jpg", img2)
                break

    return

def RRT(img, img2, start, end, stepSize):
    h,l= img.shape # dim of the loaded image
    # print(img.shape) # (384, 683)
    # print(h,l)

    # insert the starting point in the node class
    # node_list = [0] # list to store all the node points         
    node_list[0] = Nodes(start[0],start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    # display start and end
    cv2.circle(img2, (start[0],start[1]), 5,(0,0,255),thickness=3, lineType=8)
    cv2.circle(img2, (end[0],end[1]), 5,(0,0,255),thickness=3, lineType=8)

    i=1
    pathFound = False
    while pathFound==False:
        nx,ny = rnd_point(h,l)
        print("Random points:",nx,ny)

        nearest_ind = nearest_node(nx,ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        print("Nearest node coordinates:",nearest_x,nearest_y)

        #check direct connection
        tx,ty,directCon,nodeCon = check_collision(nx,ny,nearest_x,nearest_y,img)
        print("Check collision:",tx,ty,directCon,nodeCon)

        if directCon and nodeCon:
            print("Node can connect directly with end")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)

            cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (end[0],end[1]), (255,0,0), thickness=2, lineType=8)

            print("Path has been found")
            #print("parent_x",node_list[i].parent_x)
            for j in range(len(node_list[i].parent_x)-1):
                cv2.line(img2, (int(node_list[i].parent_x[j]),int(node_list[i].parent_y[j])), (int(node_list[i].parent_x[j+1]),int(node_list[i].parent_y[j+1])), (255,0,0), thickness=2, lineType=8)
            # cv2.waitKey(1)
            cv2.imwrite("media/"+str(i)+".jpg",img2)
            cv2.imwrite("out.jpg",img2)
            break

        elif nodeCon:
            print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            # print(i)
            # print(node_list[nearest_ind].parent_y)
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            i=i+1
            # display
            cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            cv2.imwrite("media/"+str(i)+".jpg",img2)
            cv2.imshow("sdc",img2)
            cv2.waitKey(1)
            continue

        else:
            print("No direct con. and no node con. :( Generating new rnd numbers")
            continue
        
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