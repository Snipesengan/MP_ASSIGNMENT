#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import HLD_Helper as imghelp
import sys

MEDIAN_BLUR_KSIZE = 21
CUTOFF_VAL = 100

def find_region_of_interest(imgray,display=False):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    res,contours,hierachy = imghelp.find_contours(imgray,mask=None)
    rects = imghelp.filter_rectangles(contours)


    black = imgray & 0
    mask = black.copy()
    #get the largest rectangle
    for i in range(3):
        rect = cv2.minAreaRect(rects[i]) #rects = (center(x,y),(width,height),angle of rotation)

        #ok now lets form a mask based on that image
        #First gets a black img that has the same shape as input image then create the mask

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = cv2.drawContours(black,[box],0,255,thickness=-1) + mask


    if display:
        roi = cv2.bitwise_and(imgray,imgray,mask=mask)
        plt.figure(str(rect))
        plt.imshow(roi,cmap='gray')
        plt.show()

    return mask,rect

#Imports a region of interest and
def identify_text(roiBGR,display=False):
    pass



def main():
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:

        imgBGR = cv2.imread(sys.argv[1])
        imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

        #Median blur effect is global --- for now. Meaning the resulting image will be used
        median = cv2.medianBlur(imgray,MEDIAN_BLUR_KSIZE)

        thresh = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,
                                       11,2)
        #find_region_of_interest(imgray,display=True)
        #find_region_of_interest(median,display=True)
        find_region_of_interest(thresh,display=True)


if __name__ == '__main__':
    main()
