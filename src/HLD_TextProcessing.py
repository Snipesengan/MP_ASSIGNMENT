import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

def find_MSER(imgray,minArea,maxArea):

    mser = cv2.MSER_create()
    mser.setMaxArea(maxArea)
    mser.setMinArea(minArea)
    mser.setDelta(250)

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

def perform_adaptive_thresh(imgBGR,threshBlock,threshC):
    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(imgHSV)
    #Adaptive gaussian on v channel
    vThresh = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,
                                    threshBlock,threshC)

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

def filter_regions_by_yCluster(regions,minY,maxY,res):

    filtered = []
    numBins = res

    histArr = []

    for r in regions:
        y = r[r[:,1].argmin()][1]
        histArr.append(y)

    hist,binEdge = np.histogram(histArr,bins=numBins,range=(minY,maxY))
    for i in np.where(hist >= 3)[0]:
        dymin = binEdge[i]
        dymax = binEdge[i + 1]

        cluster = []
        for r in regions:
            y = r[r[:,1].argmin()][1]
            if y >= dymin and y <= dymax:
                j = 0
                uniqueRegion = True

                while j <  len(filtered) and uniqueRegion == True:
                    uniqueRegion = _does_not_overlap(r,filtered[j])
                    j = j + 1

                if uniqueRegion:
                    cluster.append(r)

        filtered.append(cluster)

    return filtered

def filter_regions_by_textHomogeneity(regions,minHeight,maxHeight,res):
    filtered = []
    numBins = res

    histArr = []

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        histArr.append(h)

    hist,binEdge = np.histogram(histArr,bins=numBins,range=(minHeight,maxHeight))
    for i in np.where(hist >= 3)[0]:

        if i - 2 >= 0:
            minH = binEdge[i-2]
        else:
            minH = binEdge[0]

        if i + 2 < len(binEdge):
            maxH = binEdge[i + 2]
        else:
            maxH = binEdge[-1]

        cluster = []
        for r in regions:
            x,y,w,h = cv2.boundingRect(r)
            if h >= minH and h <= maxH:
                j = 0
                uniqueRegion = True

                while j <  len(filtered) and uniqueRegion == True:
                    uniqueRegion = _does_not_overlap(r,filtered[j])
                    j = j + 1

                if uniqueRegion:
                    cluster.append(r)

        filtered.append(cluster)

    return filtered

def _does_not_overlap(region,regions):

    notOverlap = True
    (x1,y1),(w1,h1),_ = cv2.minAreaRect(cv2.convexHull(region))
    for i,r in enumerate(regions):
        (x2,y2),(w2,h2),_ = cv2.minAreaRect(cv2.convexHull(r))

        if abs(x1 - x2) < 5 and abs(y1 - y2) < 5:
            notOverlap = False

    return notOverlap
