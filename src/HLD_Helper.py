#HDL_Helper contains useful functions and class to identify Hazmat Label

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import sys

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
        plt.figure("find_rectangles")
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()

    return rects

def calculate_color_percentage(imgBGR,mask=None):
    """Defines thresholds for colors in HSV color space"""

    #For now these are color we expect our hazmat label to have
    COLOR_BLUE   = ('blue',(98,109),(112,256))
    COLOR_GREEN  = ('green',(36,0),(74,256))
    COLOR_ORANGE = ('orange',(10,50),(15,256))
    COLOR_YELLOW = ('yellow',(20,190),(30,256))
    COLOR_RED1   = ('red',(0,70),(10,256))
    COLOR_RED2   = ('red',(170,70),(180,256))
    COLOR_WHITE  = ('white',(0,0),(180,10))

    COLOR_LIST   = [COLOR_BLUE,COLOR_GREEN,COLOR_ORANGE,COLOR_YELLOW,COLOR_RED1,COLOR_RED2]

    #A dictionary containing color as key and percentage of that color in the image as value
    colorpercentage = {}

    histnorm = _compute_2d_histgoram(imgBGR,mask)

    for c in COLOR_LIST:
        #Calculate the percentage of each color in COLOR_LIST
        #by segmenting their respective (h,s) channel range in the 2d histogram
        colorpercentage[c[0]] = np.sum(histnorm[c[1][0]:c[2][0],c[1][1]:c[2][1]])

    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    #Black and white detection are done seperately because they require
    #the V channel...
    #1.First a mask is calculated using cv.inRange() - which checks
    #   if the image is in the specify range for however many channels
    #2.Count how many non-zeros in the mask which represents hits

    #Detecting black
    black  = cv2.inRange(imgHSV,np.array([0,0,0]),np.array([180,255,0]))
    black  = np.bitwise_and(black,mask)
    count = (black != 0).sum()
    colorpercentage['black'] = count/float((mask != 0).sum())

    #Detecting white
    white  = cv2.inRange(imgHSV,np.array([0,0,200]),np.array([180,70,255]))
    white = np.bitwise_and(white,mask)
    count = (white != 0).sum()
    colorpercentage['white'] = count/float((mask != 0).sum())

    #Find out how much total percentage
    colorpercentage['total'] = sum(colorpercentage.values())

    return colorpercentage

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


#Pseudo test Harness
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        imgBGR = cv2.imread(sys.argv[1])
        hist = _compute_2d_histgoram(imgBGR)
        print(calculate_color_percentage(imgBGR))
