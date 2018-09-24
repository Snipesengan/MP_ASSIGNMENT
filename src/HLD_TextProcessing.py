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
    #Adaptive gaussian on v channel
    vThresh = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,2)

    return vThresh

#This function attemps to cluster regions based ellipse, the ratio of semi major axis to minor axis
def filter_regions_by_eccentricity(regions,maxEccentricity):

    filtered = []

    for r in regions:
        (x,y),(MA,ma),_ = cv2.fitEllipse(r)
        (_,_),(_,_),angle = cv2.minAreaRect(r)
        eccentricty = (1 - MA/ma)**(0.5)

        if eccentricty < maxEccentricity and (abs(angle - 45) > 30 != abs(angle - (-45)) > 30):
            filtered.append(r)

    return filtered

def filter_overlaping_regions(regions,tolX,tolY):
    filterOut = []

    for i1,r1 in enumerate(regions):
        hull1 = cv2.convexHull(r1)
        (x1,y1),(w1,h1),_ = cv2.minAreaRect(hull1)

        for i2, r2 in enumerate(regions):
            hull2 = cv2.convexHull(r2)
            (x2,y2),(w2,h2),_ = cv2.minAreaRect(hull2)

            if abs(x1 - x2) < tolX and abs(y1 - y2) < tolY:
                a1 = w1 * h1
                a2 = w2 * h2
                if i1 != i2:
                    if (not i1 in filterOut) and a1 > a2:
                        filterOut.append(i1)
                    elif (not i2 in filterOut) and a2 > a1:
                        filterOut.append(i2)


    filterIndices = np.delete(np.arange(len(regions),dtype=int),filterOut)

    return [regions[i] for i in filterIndices]

def filter_regions_by_yCluster(regions,minY,maxY):

    filtered = []
    numBins = 20

    histArr = []
    #filter by text size
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        histArr.append(y)

    hist,binEdge = np.histogram(histArr,bins=numBins,range=(minY,maxY))

    for i in np.where(hist >= 3)[0]:
        if abs(i - hist.argmax()) < 10: # less than 4 cluster away
            minY = binEdge[i]
            maxY = binEdge[i + 1]

            for r in regions:
                x,y,w,h = cv2.boundingRect(r)
                if y >= minY and y <= maxY:
                    filtered.append(r)

    return filtered

def filter_regions_by_textHomogeneity(regions,dy):
    filtered = []
    numBins = 30

    histArr = []
    #filter by text size
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        histArr.append(h)

    hist,binEdge = np.histogram(histArr,bins=numBins)
    print(hist)
    for i in np.where(hist >= 3)[0]:

        minHeight = binEdge[i] * (1 - dy)
        maxHeight = binEdge[i] * (1 + dy)

        for r in regions:
            x,y,w,h = cv2.boundingRect(r)
            if h >= minHeight and h <= maxHeight:
                filtered.append(r)

    return filtered
