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
import HLD_TextProcessing as textproc
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

def find_text(regions,imgBinary):

    mask = np.zeros(imgBinary.shape,np.uint8)
    cv2.drawContours(mask,regions,-1,255,-1)
    whiteText = cv2.bitwise_and(imgBinary,imgBinary,mask=mask)
    blackText = cv2.bitwise_and(255-imgBinary,255-imgBinary,mask=mask)

    if(np.bincount(whiteText.flatten())[-1] > np.bincount(blackText.flatten())[-1]):
        text = whiteText
    else:
        text = blackText

    config = ('-l eng --oem 3 --psm 7')
    cv2.imwrite("test.png",255 - text)

    return pytesseract.image_to_string(Image.open('test.png'),config=config),text

def extract_hazard_label_text_region(roiBGR,tuner):

    minBlobArea = tuner.minBlobArea
    maxBlobArea = tuner.maxBlobArea
    threshBlock = tuner.threshBlock
    threshC     = tuner.threshC

    vThresh = textproc.perform_adaptive_thresh(roiBGR)
    mserRegion,mserVis = textproc.find_MSER(vThresh,minBlobArea,maxBlobArea,threshBlock,threshC)
    filtered = textproc.filter_regions_by_eccentricity(mserRegion,tuner.maxE)
    clusterOfYRegions = textproc.filter_regions_by_yCluster(filtered,0,vThresh.shape[0])

    textVis = np.zeros(roiBGR.shape[:-1],np.uint8)
    for yRegions in clusterOfYRegions:
        clusterOfHomoRegions = textproc.filter_regions_by_textHomogeneity(yRegions,2,125,0.25)
        for homoRegions in clusterOfHomoRegions:
            textString,vis = find_text(homoRegions,vThresh)
            textVis = textVis + vis
            print(textString)



    return mserRegion, np.vstack((mserVis,255 - textVis))

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
        textROI = imgROI[tuner.textCropY:-tuner.textCropY,tuner.textCropX:-tuner.textCropX,...]


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
