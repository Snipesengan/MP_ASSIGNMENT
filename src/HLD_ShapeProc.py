#HDL_Helper contains useful functions and class to identify Hazmat Label

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

#This file contains useful functions, essentially wrapper for
#functions that already exists in opencv. However this is so
#that thresholds can be set easier this way..
def filter_rectangles(contours,epsilon):
    rects = []

    #Filter out contours that aren't rectangles
    for c in contours:
        peri = cv2.arcLength(c,True)*epsilon #The higher the number, the more rectangle it looks
        approx = cv2.approxPolyDP(c,peri,True)

        if len(approx) == 4:
            rects.append(approx)

    return rects

def filter_contour_area(contours,minArea,maxArea):

    filtered = None
    if minArea != None and maxArea != None:
        filtered = [c for c in contours if (cv2.contourArea(c) >= minArea and cv2.contourArea(c) <= maxArea)]
    elif minArea != None:
        filtered = [c for c in contours if cv2.contourArea(c) >= minArea]
    elif maxArea != None:
        filtered = [c for c in contours if cv2.contourArea(c) <= maxArea]

    return filtered

def filter_overlaping_contour(contours):

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


def find_contours(imgray,cannyMin,cannyMax,morphKernel,mask=None):

    imgray = cv2.bitwise_and(imgray,imgray,mask=mask)
    canny = cv2.Canny(imgray,cannyMin,cannyMax)
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, morphKernel)
    res,contours,hierachy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    return res,contours,hierachy
