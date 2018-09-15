#HDL_Helper contains useful functions and class to identify Hazmat Label

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

COLOR_BLUE_LOWER, COLOR_BLUE_UPPER     = np.array([98,109,20]),np.array([112,255,255])
COLOR_GREEN_LOWER, COLOR_GREEN_UPPER   = np.array([36,0,0]),np.array([75,255,255])
COLOR_YELLOW_LOWER, COLOR_YELLOW_UPPER = np.array([20,190,20]),np.array([30,255,255])
COLOR_ORANGE_LOWER, COLOR_ORANGE_UPPER = np.array([5,50,50]),np.array([15,255,255])
COLOR_RED1_LOWER, COLOR_RED1_UPPER     = np.array([0,70,50]),np.array([10,255,255])
COLOR_RED2_LOWER, COLOR_RED2_UPPER     = np.array([170,70,50]),np.array([180,255,255])
COLOR_BLACK_LOWER, COLOR_BLACK_UPPER   = np.array([0,0,0]),np.array([180,255,15])
COLOR_WHITE_LOWER, COLOR_WHITE_UPPER   = np.array([0,0,200]),np.array([180,70,255])

MIN_RECT_AREA  = 10000 #TODO: Adjust this dynamically

#This file contains useful functions, essentially wrapper for
#functions that already exists in opencv. However this is so
#that thresholds can be set easier this way..
def filter_rectangles(contours):
    rects = []

    #Filter out contours that aren't rectangles
    for c in contours:
        peri = cv2.arcLength(c,True)*0.11 #The higher the number, the more rectangle it looks
        approx = cv2.approxPolyDP(c,peri,True)

        if len(approx) == 4:
            rects.append(approx)

    filtered1_rects = _filter_contour_area(rects,MIN_RECT_AREA,None)
    filtered2_rects = _filter_overlaping_contour(filtered1_rects)

    return filtered2_rects

def find_contours(imgray,mask=None):

    imgray = cv2.bitwise_and(imgray,imgray,mask=mask)
    median = cv2.medianBlur(imgray,3)
    blurred = cv2.GaussianBlur(median,(5,5),0) #GaussianBlur(src,ksize,sigmaX)
    canny = cv2.Canny(blurred,50,150)
    #dilate = cv2.dilate(canny,np.ones((7,7),np.uint8),iterations = 1)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    res,contours,hierachy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    vis =  np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(vis,contours,-1,255,1)
    cv2.imwrite("canny.jpg",canny)
    cv2.imwrite("closing.jpg",closing)
    cv2.imwrite("contours.jpg",vis)

    return res,contours,hierachy

def calculate_color_percentage(imgBGR,mask=None,display=False):

    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)
    colormap = {}

    #Calculate percentage of each color
    colormap['blue']   = _calculate_color_percent(imgHSV,COLOR_BLUE_LOWER,COLOR_BLUE_UPPER,mask)
    colormap['green']  = _calculate_color_percent(imgHSV,COLOR_GREEN_LOWER,COLOR_GREEN_UPPER,mask)
    colormap['yellow'] = _calculate_color_percent(imgHSV,COLOR_YELLOW_LOWER,COLOR_YELLOW_UPPER,mask)
    colormap['orange'] = _calculate_color_percent(imgHSV,COLOR_ORANGE_LOWER,COLOR_ORANGE_UPPER,mask)
    colormap['black']  = _calculate_color_percent(imgHSV,COLOR_BLACK_LOWER,COLOR_BLACK_UPPER,mask)
    colormap['white']  = _calculate_color_percent(imgHSV,COLOR_WHITE_LOWER,COLOR_WHITE_UPPER,mask)

    #For red, since there are two thresholds - need to added them together
    red1 = _calculate_color_percent(imgHSV,COLOR_RED1_LOWER,COLOR_RED1_UPPER,mask)
    red2 = _calculate_color_percent(imgHSV,COLOR_RED2_LOWER,COLOR_RED2_UPPER,mask)
    red_mask  = np.clip(red1[1] + red2[1],0,255)
    red_percent = red1[0] + red2[0]

    colormap['red'] = red_percent,red_mask

    #Sort the colors cuz why not
    sortcolor = list(colormap.keys())
    sortcolor.sort(key = lambda x: colormap[x][0],reverse=True)
    sortedmap = {key: colormap[key] for key in sortcolor}

    #Display the most dominant color
    if display:
        color,v = list(sortedmap.items())[0]
        color_mask = v[1]
        img = cv2.bitwise_and(imgBGR,imgBGR,mask=color_mask)
        plt.figure("Dominant Color: " + color)
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()

    return sortedmap #key: color, value: (colorpercentage,colormask)


def find_MSER(imgBGR,mask=None,display=False):
    mser = cv2.MSER_create()

    imgBGR = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)
    gray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

    regions, _ = mser.detectRegions(gray)

    if display:
        hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]
        vis = imgBGR.copy()
        cv2.polylines(vis,hulls,1,(0,255,0),thickness=3)
        plt.figure("MSER")
        plt.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB))

    return regions


#Calculate the percentage of that color and its mask
def _calculate_color_percent(imgHSV,lower,upper,mask=None):

    if mask.any != None:
        pixel_count = (mask != 0).sum()
    else:
        pixel_count = imgHSV.shape[0] * imgHSV.shape[1]

    color = cv2.inRange(imgHSV,lower,upper)
    color = cv2.bitwise_and(color,color,mask=mask)
    color_count = (color != 0).sum()

    return (float(color_count)/pixel_count, color)

def _filter_contour_area(contours,minArea,maxArea):

    filtered = None
    if minArea != None and maxArea != None:
        filtered = [c for c in contours if (cv2.contourArea(c) >= minArea and cv2.contourArea(c) <= maxArea)]
    elif minArea != None:
        filtered = [c for c in contours if cv2.contourArea(c) >= minArea]
    elif maxArea != None:
        filtered = [c for c in contours if cv2.contourArea(c) <= maxArea]

    return filtered

def _filter_overlaping_contour(contours):

    filtered = []

    #sort the contours by area - largest to smallest
    contours.sort(key=lambda x: cv2.contourArea(x),reverse=True)

    for c1 in contours:
        (x,y),_,_ = cv2.minAreaRect(c1)
        overlapping = False

        for c2 in filtered:
            leftmost = tuple(c2[c2[:,:,0].argmin()][0])
            rightmost = tuple(c2[c2[:,:,0].argmax()][0])
            topmost = tuple(c2[c2[:,:,1].argmin()][0])
            bottommost = tuple(c2[c2[:,:,1].argmax()][0])

            if int(x) > leftmost[0] and int(x) < rightmost[0] and int(y) > topmost[1] and y < bottommost[1]:
                overlapping = True
                break

        if not overlapping:
            filtered.append(c1)

    return filtered
