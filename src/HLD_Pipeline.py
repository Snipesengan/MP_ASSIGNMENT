#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import HLD_Helper as imghelp
import sys

def find_region_of_interest(imgBGR,display=False):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label.

    #blur the image
    rects = imghelp.find_rectangles(imgBGR,display=display)

    #get the largest rectangle
    rect = cv2.minAreaRect(rects[0]) #rects = (center(x,y),(width,height),angle of rotation)

    #ok now lets form a mask based on that image
    #First gets a black img that has the same shape as input image then create the mask
    black = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY) & 0
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    mask = cv2.drawContours(black,[box],0,255,thickness=-1)
    if display:
        roi = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
        plt.figure("find_region_of_interest")
        plt.imshow(roi)
        plt.show()

    return mask

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        imgBGR = cv2.imread(sys.argv[1])
        mask = find_region_of_interest(imgBGR,True)
        colorpercentage = imghelp.calculate_color_percentage(imgBGR,mask)
        print(colorpercentage)
