import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

def find_MSER(imgray,minArea,maxArea,blockSize,C):

    mser = cv2.MSER_create()
    mser.setMaxArea(maxArea)
    mser.setMinArea(minArea)

    regions, _ = mser.detectRegions(imgray)

    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]

    areaFilter = []
    for i,region in enumerate(hulls):
        area = cv2.contourArea(region)
        if area >= minArea and area <= maxArea:
            areaFilter.append(regions[i])

    vis = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(vis,[cv2.convexHull(p.reshape(-1,1,2)) for p in areaFilter],-1,255,2)

    return areaFilter,vis

def perform_adaptive_thresh(imgBGR):
    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(imgHSV)
    #Adaptive gaussian on s channel
    vThresh = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,2)

    return vThresh

#This function attemps to cluster regions based ellipse, the ratio of semi major axis to minor axis
def filter_regions_by_eccentricity(regions,maxEccentricity):

    filtered = []

    for r in regions:
        (x,y),(MA,ma),angle = cv2.fitEllipse(r)
        eccentricty = (1 - MA/ma)**(0.5)

        if eccentricty < maxEccentricity:
            filtered.append(r)

    return filtered

def filter_regions_by_yCluster(regions):

    filtered = []
    numBins = 4

    histArr = []
    #filter by text size
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        histArr.append(y)

    hist,binEdge = np.histogram(histArr,bins=numBins)
    print(hist,binEdge)
    for i in np.where(hist >= 3)[0]:
        minY = binEdge[i]
        maxY = binEdge[i + 1]

        for r in regions:
            x,y,w,h = cv2.boundingRect(r)
            if y >= minY and y <= maxY:
                filtered.append(r)

    return filtered

def filter_regions_by_textHomogeneity(regions):
    filtered = []
    numBins = 6

    histArr = []
    #filter by text size
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        histArr.append(h)

    hist,binEdge = np.histogram(histArr,bins=numBins)
    minHeight = binEdge[hist.argmax()]
    maxHeight = binEdge[hist.argmax() + 1]

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        if h >= minHeight and h <= maxHeight:
            filtered.append(r)


    return filtered
