#HDL_Helper contains useful functions and class to identify Hazmat Label

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import sys

COLOR_BLUE_LOWER, COLOR_BLUE_UPPER     = np.array([98,109,20]),np.array([112,255,255])
COLOR_GREEN_LOWER, COLOR_GREEN_UPPER   = np.array([36,0,0]),np.array([75,255,255])
COLOR_YELLOW_LOWER, COLOR_YELLOW_UPPER = np.array([20,190,20]),np.array([30,255,255])
COLOR_ORANGE_LOWER, COLOR_ORANGE_UPPER = np.array([5,50,50]),np.array([15,255,255])
COLOR_RED1_LOWER, COLOR_RED1_UPPER     = np.array([0,70,50]),np.array([10,255,255])
COLOR_RED2_LOWER, COLOR_RED2_UPPER     = np.array([170,70,50]),np.array([180,255,255])
COLOR_BLACK_LOWER, COLOR_BLACK_UPPER   = np.array([0,0,0]),np.array([180,255,15])
COLOR_WHITE_LOWER, COLOR_WHITE_UPPER   = np.array([0,0,200]),np.array([180,70,255])




#Takes in a region of interest in of the image
def localize_text_in_image(roiBGR,display=False):
    #Ok how do we do this lmao


#This file contains useful functions, essentially wrapper for
#functions that already exists in opencv. However this is so
#that thresholds can be set easier this way..
def find_rectangles(imgBGR,mask=None,display=False):

    rects = []

    if(mask != None):
        imgray = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)

    #Find the contours in the image
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray,(5,5),1)
    canny = cv2.Canny(blurred,100,200)
    #Perform some morphology on canny to make edge bigger
    dilate = cv2.dilate(canny,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    res,contours,hierachy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Filter out rectangles in the contours
    img = imgBGR.copy()
    for c in contours:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.04*peri,True)

        if len(approx) == 4:
            cv2.drawContours(img,[c],-1,(0,255,0),2)
            rects.append(c)
            area = cv2.contourArea(c)

    #Sort the rects based on area, largest area first
    rects.sort(key = lambda x: cv2.contourArea(x),reverse=True)

    if display:
        plt.figure("Canny")
        plt.imshow(canny,cmap='gray')
        plt.figure("find_rectangles")
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()


    return rects

def calculate_color_percentage(imgBGR,mask=None,display=False):

    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)
    colormap = {}

    #Calculate percentage of each color
    colormap['blue']   = _calculate_percent(imgHSV,COLOR_BLUE_LOWER,COLOR_BLUE_UPPER,mask)
    colormap['green']  = _calculate_percent(imgHSV,COLOR_GREEN_LOWER,COLOR_GREEN_UPPER,mask)
    colormap['yellow'] = _calculate_percent(imgHSV,COLOR_YELLOW_LOWER,COLOR_YELLOW_UPPER,mask)
    colormap['orange'] = _calculate_percent(imgHSV,COLOR_ORANGE_LOWER,COLOR_ORANGE_UPPER,mask)
    colormap['black']  = _calculate_percent(imgHSV,COLOR_BLACK_LOWER,COLOR_BLACK_UPPER,mask)
    colormap['white']  = _calculate_percent(imgHSV,COLOR_WHITE_LOWER,COLOR_WHITE_UPPER,mask)

    #For red, since there are two thresholds - need to added them together
    red1 = _calculate_percent(imgHSV,COLOR_RED1_LOWER,COLOR_RED1_UPPER,mask)
    red2 = _calculate_percent(imgHSV,COLOR_RED2_LOWER,COLOR_RED2_UPPER,mask)
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

#Calculate the percentage of that color and its mask
def _calculate_percent(imgHSV,lower,upper,mask=None):

    if mask.any != None:
        pixel_count = (mask != 0).sum()
    else:
        pixel_count = imgHSV.shape[0] * imgHSV.shape[1]

    color = cv2.inRange(imgHSV,lower,upper)
    color = cv2.bitwise_and(color,color,mask=mask)
    color_count = (color != 0).sum()

    return (float(color_count)/pixel_count, color)

#Imports a img in BGR color space
def _compute_2d_histgoram(imgBGR,mask=None):
    hsv = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    #calcHist():
    #   img      = input image, for this it should be converted to HSV
    #   channels = [0,1], process both H and S plane
    #   mask     = no mask yet.... will later when we figure out how to find region of interest
    #   bins     = [180,256], 180 for H and 256 for S plane
    #   range    = [0,180,0,256] ... self explanatory

    hist = cv2.calcHist([hsv],[0,1],mask,[180,256],[0,180,0,256])

    #now we need to normalize this hist value to be between 0 and 100 - representative of the
    #percentage in teh original image
    histnorm = hist.astype(float)/np.sum(hist)
    return histnorm
