#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import sys

import HLD_Helper as imghelp
import HLD_Transform as transform
import HLD_Tuner

def find_region_of_interest(imgray,tuner):

    cannyMin   = tuner.cannyMin
    cannyMax   = tuner.cannyMax
    morphK     = tuner.morphK
    minROIArea = tuner.minROIArea
    maxROIArea = tuner.maxROIArea
    epsilon    = tuner.epsilon

    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*
    hazardlabels_contours_mask = []
    res,contours,hierachy = imghelp.find_contours(imgray,cannyMin,cannyMax,morphK)
    contours = imghelp.filter_contour_area(contours,minROIArea,maxROIArea) #contours,minArea,maxArea
    rects = imghelp.filter_rectangles(contours,epsilon)
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
def extract_hazard_label_text_region(roiBGR,tuner):

    minBlobArea = tuner.minBlobArea
    maxBlobArea = tuner.maxBlobArea
    threshBlock = tuner.threshBlock
    threshC     = tuner.threshC

    roiGray = cv2.cvtColor(roiBGR,cv2.COLOR_BGR2GRAY)
    mserRegion,mserVis,thresh = imghelp.find_MSER(roiGray,minBlobArea,maxBlobArea,threshBlock,threshC)
    filtered = imghelp.filter_regions_by_eccentricity(mserRegion,tuner.maxE)

    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in filtered]
    mask = np.zeros(thresh.shape,np.uint8)
    cv2.fillPoly(mask,hulls,255)

    textBinary = cv2.bitwise_and(roiGray,roiGray,mask=mask)

    cv2.imwrite("TesseractStoreImg.png",textBinary)
    config = ('-l eng --oem 1 --psm 6')
    text = pytesseract.image_to_string(Image.open("TesseractStoreImg.png"),config=config)
    print(text)

    return mserRegion, np.vstack((thresh,mserVis,textBinary))

#The main pipe line
def run_detection(imgpath,display):

    ROIList = []
    mserRegionList = []
    tuner = HLD_Tuner.Tuner()

    if display:
        mserVisList = []
        roiVisList = []

    imgBGR = cv2.imread(imgpath)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(imgray,tuner.medianKSize)
    blurred = cv2.GaussianBlur(median,tuner.gaussKSize,tuner.gaussSigmaX)
    hl_c_m = find_region_of_interest(blurred,tuner)

    for i, (rectContour,mask) in enumerate(hl_c_m):
        imgROI = transform.perspective_trapezoid_to_rect(imgBGR,rectContour,tuner.finalSize,mask)
        ROIList.append(imgROI)

        #Lets crop the imgROI into thirds

        textROI = imgROI[int(imgROI.shape[1]*1/3):int(imgROI.shape[1]*2/3),50:-50,...]

        mserRegion,textROIVis = extract_hazard_label_text_region(textROI,tuner)
        mserRegionList.append(mserRegion)
        if display:
            roiVisList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))
            mserVisList.append(textROIVis)

    if display:
        plt.figure("Hazard Label Detection")
        plt.subplot(311)
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(ROIList) > 0:
            plt.subplot(312)
            plt.imshow(np.hstack(tuple(roiVisList)))
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
