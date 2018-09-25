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
import HLD_Misc as imgmisc

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

def find_text(textImg):
    config = ('-l eng --oem 1 --psm 7')
    cv2.imwrite("test.png",255 - textImg)

    return pytesseract.image_to_string(Image.open('test.png'),config=config)

def extract_hazard_label_text(roiBGR,tuner):

    text = ""

    minBlobArea   = tuner.minBlobArea
    maxBlobArea   = tuner.maxBlobArea
    blobDelta    = tuner.blobDelta
    threshBlock   = tuner.threshBlock
    threshC       = tuner.threshC
    minTextHeight = tuner.minTextHeight
    maxTextHeight = tuner.maxTextHeight
    textHRes      = tuner.textHRes
    minY          = tuner.minTextY
    maxY          = tuner.maxTextY
    yRes          = tuner.textYRes

    vThresh = textproc.perform_adaptive_thresh(roiBGR,threshBlock,threshC)
    mserRegion,mserVis = textproc.find_MSER(vThresh,minBlobArea,maxBlobArea,blobDelta)
    filtered = textproc.filter_regions_by_eccentricity(mserRegion,tuner.maxE)
    yCluster = textproc.filter_regions_by_yCluster(filtered,minY,maxY,yRes)

    textVis = np.zeros(vThresh.shape)
    for regions1 in yCluster:
        homoCluster = textproc.filter_regions_by_textHomogeneity(regions1,minTextHeight,maxTextHeight,
                                                                 textHRes)
        for regions2 in homoCluster:
            solidity = imgmisc.calculate_solidity(regions2)
            if solidity > 2:
                textImg = textproc.space_out_text(vThresh,regions2)
                textTmp = find_text(textImg)
                text = text + textTmp
                textVis = cv2.drawContours(textVis,regions2,-1,255,-1)

        text = text + '\n'

    return text, np.hstack((vThresh,mserVis,textVis))

#The main pipe line
def run_detection(imgpath,display):

    ROIList = []
    textFoundList = []
    tuner = HLD_Tuner.Tuner()

    if display:
        textVisList = []
        roiVisList = []

    imgBGR = cv2.imread(imgpath)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(imgray,tuner.medianKSize)
    blurred = cv2.GaussianBlur(median,tuner.gaussKSize,tuner.gaussSigmaX)
    hl_c_m = find_region_of_interest(blurred,tuner)

    for i, (rectContour,mask) in enumerate(hl_c_m):
        imgROI = transform.perspective_trapezoid_to_rect(imgBGR,rectContour,tuner.finalSize,mask)
        ROIList.append(imgROI)

        text,textVis = extract_hazard_label_text(imgROI,tuner)
        print(text)
        textFoundList.append(text)

        if display:
            roiVisList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))
            textVisList.append(textVis)

    #now sanity check the text list


    if display:
        plt.figure("Hazard Label Detection")
        plt.subplot(311)
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(ROIList) > 0:
            plt.subplot(312)
            plt.imshow(np.hstack(tuple(roiVisList)))
            plt.subplot(313)
            plt.imshow(np.hstack(tuple(textVisList)),cmap='gray')
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
