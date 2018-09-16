#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import HLD_Helper as imghelp
import pytesseract
import sys
from PIL import Image
import Transform


def find_region_of_interest(imgray):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    hazardlabels_contours_mask = []
    res,contours,hierachy = imghelp.find_contours(imgray,30,150,np.ones((7,7),np.uint8))
    contours = imghelp.filter_contour_area(contours,10000,None) #contours,minArea,maxArea
    rects = imghelp.filter_rectangles(contours)
    rects = imghelp.filter_overlaping_contour(rects)

    #For each rect contours, create a the corresponding mask
    black = np.zeros(imgray.shape,np.uint8)
    displayMask = []
    for rectContour in rects:
        mask = np.zeros(imgray.shape,np.uint8)
        cv2.fillPoly(mask,[rectContour],255)
        hazardlabels_contours_mask.append((rectContour,mask))
        displayMask.append(mask)

    return hazardlabels_contours_mask

#Use MSER to extract
def extract_hazard_label_text_region(roiBGR,medianKsize,gaussK,threshBSize,threshC):
    roiGray = cv2.cvtColor(roiBGR,cv2.COLOR_BGR2GRAY)
    roiMedian = cv2.medianBlur(roiGray,medianKsize)
    roiBlurred = cv2.GaussianBlur(roiMedian,gaussK,0)
    roiThresh = cv2.adaptiveThreshold(roiBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,
                                      threshBSize,threshC)
    mserRegion,mserVis = imghelp.find_MSER(roiThresh)

    return mserRegion,mserVis
#The main pipe line
def run_detection(imgpath,display):

    ROIList = []
    mserRegionList = []

    if display:
        mserVisList = []

    imgBGR = cv2.imread(imgpath)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(imgray,3)
    blurred = cv2.GaussianBlur(median,(5,5),0) #GaussianBlur(src,ksize,sigmaX)
    hl_c_m = find_region_of_interest(blurred)

    for i, (rectContour,mask) in enumerate(hl_c_m):
        imgROI = Transform.perspective_trapezoid_to_rect(imgBGR,rectContour,mask)
        ROIList.append(imgROI)

    for roi in ROIList:
        mserRegion,mserVis = extract_hazard_label_text_region(roi,5,(5,5),17,2)
        mserRegionList.append(mserRegion)
        if display:
            mserVisList.append(mserVis)

    if display:
        plt.figure("Hazard Label Detection")
        plt.subplot(311)
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(ROIList) > 0:
            plt.subplot(312)
            roiVis = [cv2.cvtColor(img,cv2.COLOR_BGR2RGB) for img in ROIList]
            plt.imshow(np.hstack(tuple(roiVis)))
            plt.subplot(313)
            plt.imshow(np.hstack(tuple(mserVisList)),cmap='gray')
        else:
            print("No ROI found")

        plt.show()

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        print("Usage -- python {script} <image_path> <-display>".format(script=sys.argv[0]))
    else:
        imgpath = sys.argv[1]
        display = False
        if len(sys.argv) == 3 and sys.argv[2] == '-display':
            display = True

        run_detection(imgpath,display)
