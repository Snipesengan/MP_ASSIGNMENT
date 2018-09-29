import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import sys
import math
import re


COLOR_RED1_LOWER, COLOR_RED1_UPPER     = np.array([0,35,55]),np.array([4,255,255])
COLOR_RED2_LOWER, COLOR_RED2_UPPER     = np.array([170,35,55]),np.array([180,255,255])
COLOR_ORANGE_LOWER, COLOR_ORANGE_UPPER = np.array([7,60,50]),np.array([19,255,255])
COLOR_YELLOW_LOWER, COLOR_YELLOW_UPPER = np.array([16,120,20]),np.array([30,255,255])
COLOR_BLUE_LOWER, COLOR_BLUE_UPPER     = np.array([110,109,20]),np.array([120,255,255])
COLOR_GREEN_LOWER,COLOR_GREEN_UPPER     = np.array([23,0,0]),np.array([85,255,255])
COLOR_BLACK_LOWER, COLOR_BLACK_UPPER   = np.array([0,0,0]),np.array([180,255,60])
COLOR_WHITE_LOWER, COLOR_WHITE_UPPER   = np.array([0,0,90]),np.array([180,90,255])

def color_correction(imgBGR):

    #Increasing Saturation to better detect color
    """
    imgHSV    = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)
    H,S,V     = cv2.split(imgHSV)

    SEq       = cv2.equalizeHist(S)
    imgBGR    = cv2.cvtColor(cv2.merge((H,SEq,V)),cv2.COLOR_HSV2BGR)
    """
    #White balancing

    imgYCrCb  = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb   = cv2.split(imgYCrCb)
    YEq       = cv2.equalizeHist(Y)
    imgBGR    = cv2.cvtColor(cv2.merge((YEq,Cr,Cb)),cv2.COLOR_YCrCb2BGR)


    return imgBGR

def calculate_color_percentage(imgBGR,mask=None,display=False):

    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)
    colormap = {}

    #Calculate percentage of each color
    colormap['blue']   = _calculate_color_percent(imgHSV,COLOR_BLUE_LOWER,COLOR_BLUE_UPPER,mask)
    colormap['green']  = _calculate_color_percent(imgHSV,COLOR_GREEN_LOWER,COLOR_GREEN_UPPER,mask)
    colormap['yellow'] = _calculate_color_percent(imgHSV,COLOR_YELLOW_LOWER,COLOR_YELLOW_UPPER,mask)
    colormap['orange'] = _calculate_color_percent(imgHSV,COLOR_ORANGE_LOWER,COLOR_ORANGE_UPPER,mask)
    colormap['white']  = _calculate_color_percent(imgHSV,COLOR_WHITE_LOWER,COLOR_WHITE_UPPER,mask)
    colormap['black']  = _calculate_color_percent(imgHSV,COLOR_BLACK_LOWER,COLOR_BLACK_UPPER,mask)

    #For red, since there are two thresholds - need to added them together
    red1 = _calculate_color_percent(imgHSV,COLOR_RED1_LOWER,COLOR_RED1_UPPER,mask)
    red2 = _calculate_color_percent(imgHSV,COLOR_RED2_LOWER,COLOR_RED2_UPPER,mask)
    red_percent = red1 + red2
    colormap['red'] = red_percent

    #Sort the colors cuz why not
    sortcolor = list(colormap.keys())
    sortcolor.sort(key = lambda x: colormap[x],reverse=True)
    sortedmap = {key: "%.3f"%(colormap[key]*100) for key in sortcolor}

    return sortedmap #key: color, value: (colorpercentage,colormask)


#Calculate the percentage of that color and its mask
def _calculate_color_percent(imgHSV,lower,upper,mask=None):

    if type(mask) is np.ndarray:
        pixel_count = (mask != 0).sum()
    else:
        pixel_count = imgHSV.shape[0] * imgHSV.shape[1]

    color = cv2.inRange(imgHSV,lower,upper)
    color = cv2.bitwise_and(color,color,mask=mask)

    color_count = np.bincount(color.flatten(),minlength=2)[-1]
    #print(np.bincount(color.flatten(),minlength=2))
    #plt.imshow(np.hstack((imgHSV,cv2.cvtColor(cv2.bitwise_and(imgHSV,imgHSV,mask=color),cv2.COLOR_HSV2RGB))))
    #plt.show()

    return float(color_count)/pixel_count

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    color_correction(img)
