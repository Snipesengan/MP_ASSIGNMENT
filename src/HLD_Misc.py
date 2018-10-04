import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import sys
import math
import HLD_Misc as imgmisc

def display_region(regions,dim):
    mask = np.zeros(dim,np.uint8)
    cv2.drawContours(mask,regions,-1,255,1)
    plt.figure("Regions")
    plt.imshow(mask,cmap='gray')
    plt.show()

#Imports: Image, threshold values
#Exports a adaptive threshed image
def perform_adaptive_thresh(imgBGR,threshBlock,threshC,threshErode):
    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    #Adaptive thresholding is performed on the v channel of HSV to remove
    #shadow. Because we want to binarize regions with varying light intensity
    #rather than saturation or color. This is because the feature we are extracting
    #Are text and symbols which in this case does not vary much in hue but rather
    #intensity
    h,s,v = cv2.split(imgHSV)
    vThresh = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,
                                    threshBlock,threshC)

    #Apply median and erosion to remove noise
    median = cv2.medianBlur(vThresh,3)
    erode = cv2.erode(vThresh,threshErode,iterations = 1)

    return erode

#Imports MSER region and the output shape of the mask
def get_mask_from_regions(regions,shape):
    mask = np.zeros(shape,np.uint8)
    #for regions with holes in them
    cv2.drawContours(mask,regions,-1,255,1)
    return mask

def calculate_solidity(regions):
    left = []
    right = []
    top = []
    bottom = []
    regionArea = 0

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        left.append(x)
        right.append(x+w)
        top.append(y)
        bottom.append(y+h)

        regionArea = regionArea + cv2.contourArea(r)

    x1 = left[np.array(left).argmin()]
    y1 = top[np.array(top).argmin()]
    x2 = right[np.array(right).argmax()]
    y2 = bottom[np.array(bottom).argmax()]

    boxArea = (y2 - y1) * (x2 - x1)

    return (regionArea/boxArea) * 100

#calculate the centroid of the region (center point)
#this is used to sort regions from left to right
def calculate_centroid(regions):

    centroids = []

    for r in regions:
        M = cv2.moments(cv2.convexHull(r))
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            cx = 0
            cy = 0
        centroids.append(np.array([cx,cy]))

    centroids = np.array(centroids)
    centerX = np.average(centroids[:,0],axis=0)
    centerY = np.average(centroids[:,1],axis=0)

    return (centerX,centerY)
