import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math
import HLD_Misc as imgmisc

#Finds Maximally Stable Extremal Regions
def find_MSER(imgray,minArea,maxArea,delta):

    mser = cv2.MSER_create()
    mser.setMaxArea(maxArea)
    mser.setMinArea(minArea)
    mser.setDelta(delta)

    #Filter the regions based on their area
    regions, _ = mser.detectRegions(imgray)
    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]
    areaFilter = [regions[i] for i,r in enumerate(hulls) if (cv2.contourArea(r) > minArea and cv2.contourArea(r) < maxArea)]

    return areaFilter

#Calculate the total area of multiple regions
def calculate_regions_area(regions,shape):
    #Note that areas are number of pixels
    mask = np.zeros(shape,dtype=np.uint8)
    cv2.drawContours(mask,regions,-1,255,-1)
    area = np.bincount(mask.flatten(),minlength=2)[-1]

    return area


def filter_regions_by_solidty(regions,minSolidity,maxSolidity):
    filtered = []

    for r in regions:
        area = cv2.contourArea(r)
        hull = cv2.convexHull(r)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        if solidity >= minSolidity and solidity <= maxSolidity:
            filtered.append(r)

    return filtered

#filter regions by min and max area
def filter_regions_by_area(regions,minArea,maxArea):
    filtered = []

    for r in regions:
        area = cv2.contourArea(cv2.convexHull(r))
        if area >= minArea and area <= maxArea:
            filtered.append(r)

    return filtered

#filter regions by their eccentricity
def filter_regions_by_eccentricity(regions,maxEccentricity):

    filtered = []

    for r in regions:
        (x,y),(MA,ma),_ = cv2.fitEllipse(r)
        (_,_),(_,_),angle = cv2.minAreaRect(r)
        eccentricty = (1 - MA/ma)**(0.5)

        if eccentricty < maxEccentricity and (abs(angle - 45) > 15 != abs(angle - (-45)) > 15):
            filtered.append(r)

    return filtered

#filter regions by their location has to be within (x1,y1,x2,y2)
def filter_regions_by_location(regions,rect):
    x1,y1,w1,h1 = rect
    filtered = []
    for r in regions:
        (x,y),_ = cv2.minEnclosingCircle(r)
        if (x >= x1 and x <= (x1 + w1)) and (y >= y1 and y <= (y1 + h1)):
            filtered.append(r)

    return filtered

#Cluster regions together based on the similarity in height, y coord, x coord
def approx_homogenous_regions_chain(regions,deltaX,deltaY,C,minLength=2):
    getKey = lambda x: str(cv2.minEnclosingCircle(x)[0])
    visited = set()
    clusters = []
    keepgoing = True
    #This uses a depth first search algorithm that groups regions together
    while len(visited) < len(regions):
        unvisited = [r for r in regions if not getKey(r) in visited]
        unvisited.sort(key = lambda x: cv2.boundingRect(x)[3])
        cluster = []
        currNode = unvisited.pop(0)
        _DFS(currNode,regions,visited,deltaX,deltaY,C,cluster)
        if(len(cluster) >= minLength):
            clusters.append(cluster)

    return clusters

#Depth first search algorithm that queues together regions with similar height,ycoord,xcoord
def _DFS(currNode,regions,visited,deltaX,deltaY,C,cluster):
    getKey = lambda x: str(cv2.minEnclosingCircle(x)[0])
    visited.add(getKey(currNode))
    cluster.append(currNode)
    nearbys = find_Nearest_Homogenous_Regions(currNode,regions,deltaX,deltaY,C)
    for nextNode in nearbys:
        if not getKey(nextNode) in visited:
            _DFS(nextNode,regions,visited,deltaX,deltaY,C,cluster)

#Returns a lists of regions that are similar (used by _DFS)
def find_Nearest_Homogenous_Regions(currNode,regions,deltaX,deltaY,C):
    x,y,w,h = cv2.boundingRect(currNode)
    rect = (x-deltaX*w,y-deltaY*h,(2*deltaX+1)*w,(2*deltaY+1)*h)
    nearbys = filter_regions_by_location(regions,rect)
    #now find one that similarly sized
    nearbys = [r for r in nearbys if abs(cv2.boundingRect(r)[3] - h)/h < C]

    return nearbys

#Filter overlapping regions
def filter_overlapping_regions(regions):

    filtered = []

    #sort the contours by area - largest to smallest
    regions.sort(key=lambda r: cv2.contourArea(cv2.convexHull(r)),reverse=True)

    for r1 in regions:
        (x,y),_,_ = cv2.minAreaRect(r1)
        overlapping = False

        for r2 in filtered:
            leftmost = r2[r2[:,0].argmin()]
            rightmost = r2[r2[:,0].argmax()]
            topmost = r2[r2[:,1].argmin()]
            bottommost = r2[r2[:,1].argmax()]

            if int(x) > leftmost[0] and int(x) < rightmost[0] and int(y) > topmost[1] and y < bottommost[1]:
                overlapping = True
                break

        if not overlapping:
            filtered.append(r1)

    return filtered

#Sorts regions from left to right
def sort_left_to_right(cluster):
    def keyFunc(regions):
        cx,cy = imgmisc.calculate_centroid(regions)
        return math.sqrt(cx**2 + cy**2)

    tosort = cluster.copy()
    tosort.sort(key = keyFunc)
    return tosort
