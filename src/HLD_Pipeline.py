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
import math
import re

import HLD_ShapeProc as shapeproc
import HLD_TextProcessing as textproc
import HLD_RegionsProc as regionproc
import HLD_Transform as transform
import HLD_Tuner
import HLD_ColorProc as colorproc
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
    res,contours,hierachy = shapeproc.find_contours(imgray,cannyMin,cannyMax,morphK)
    contours = shapeproc.filter_contour_area(contours,minROIArea,maxROIArea) #contours,minArea,maxArea
    rects = shapeproc.filter_rectangles(contours,epsilon)
    rects = shapeproc.filter_overlaping_contour(rects)

    #For each rect contours, create a the corresponding mask
    black = np.zeros(imgray.shape,np.uint8)
    displayMask = []
    for rectContour in rects:
        mask = np.zeros(imgray.shape,np.uint8)
        cv2.fillPoly(mask,[rectContour],255)
        hazardlabels_contours_mask.append((rectContour,mask))
        displayMask.append(mask)

    return hazardlabels_contours_mask

def find_text(textImg,config='-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
    cv2.imwrite("test.png",255 - textImg)

    return pytesseract.image_to_string(Image.open('test.png'),config=config)

def extract_hazard_label_text(roiBGR,tuner):

    text = ""
    classNumber = ""
    dictionary = open("dictionary.txt").read().split('\n')
    textVis = []

    vThresh    = imgmisc.perform_adaptive_thresh(roiBGR,tuner.threshBlock,tuner.threshC)
    mserRegion = regionproc.find_MSER(vThresh,tuner.minBlobArea,tuner.maxBlobArea,tuner.blobDelta)

    filtered   = regionproc.filter_regions_by_eccentricity(mserRegion,tuner.maxE)
    filtered   = regionproc.filter_overlapping_regions(filtered)

    #FOR LABEL
    regionCluster = regionproc.approx_homogenous_regions_chain(filtered,3,0.2,0.2,minLength=2)
    regionCluster = regionproc.sort_left_to_right(regionCluster,roiBGR.shape[0])
    for regions in regionCluster:
        space = sum([cv2.boundingRect(r)[2] for r in regions])/len(regions)
        textImg = textproc.space_out_text(roiBGR,regions,space)
        tessOut =  find_text(textImg)
        textTmp = ''.join(re.findall(r"[A-Z]|!",tessOut))
        textVis.append(regions)
        longest = ""
        for words in dictionary:
            lcs = textproc.find_LCS(textTmp,words)
            if len(lcs) > 0:
                tmp = lcs.pop()
                if len(tmp) > len(longest): longest = tmp
        if len(longest) >= 3:
            text = text + textTmp + ' '

    text = re.sub(r" (?= )","",text.strip())

    #FOR CLASS NUMBER
    #get only the bottom regions where the class number is
    classRegions = regionproc.filter_regions_by_location(filtered,(150,350,150,150))
    if len(classRegions) > 0:
        classImg = textproc.space_out_text(roiBGR,classRegions,10)
        classNumber = find_text(classImg,config='-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789')
        matches = re.findall(r"[0-9]|\.(?=[0-9])",classNumber)
        classNumber = ''.join(matches)

    #for visual stuff
    vis = cv2.cvtColor(vThresh,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(vis,[cv2.convexHull(r) for r in mserRegion],-1,(255,0,0),2)
    for i,regions in enumerate(textVis):
        color = [0,0,0]
        color[i%3] = 255

        cv2.drawContours(vis,regions,-1,tuple(color),1)

    tmp = cv2.drawContours(np.zeros(vThresh.shape,dtype=np.uint8),mserRegion,-1,255,-1)
    nonRegThresh = vThresh - tmp

    return (text,classNumber),vis,nonRegThresh

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
        roiMask = 255 - np.zeros(imgBGR.shape[:-1],dtype=np.uint8)
        imgROI = transform.perspective_trapezoid_to_rect(imgBGR,rectContour,tuner.finalSize,mask)
        roiMask = transform.perspective_trapezoid_to_rect(roiMask,rectContour,tuner.finalSize,mask)
        ROIList.append(imgROI)

        (label,classNo),textVis,nonRegThresh = extract_hazard_label_text(imgROI,tuner)
        roiMask = roiMask - (255 - nonRegThresh)
        #plt.imshow(cv2.cvtColor(imgROI,cv2.COLOR_BGR2HSV))
        topMap = colorproc.calculate_color_percentage(imgROI[0:250,...],roiMask[0:250,...])
        botMap = colorproc.calculate_color_percentage(imgROI[250:499,...],roiMask[250:499,...])
        topColor = list(topMap.keys())[0]
        botColor = list(botMap.keys())[0]
        print(topMap)
        print(botMap)

        print("TOP         : " + topColor)
        print("BOTTOM      : " + botColor)
        print("LABEL       : " + label)
        print("CLASS NUMBER: " + classNo)

        textFoundList.append(label)

        if display:
            roiVisList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))
            textVisList.append(textVis)

    #now sanity check the text list
    if display:
        plt.figure("Input Image")
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(ROIList) > 0:
            plt.figure("Detecting hazard label")
            plt.imshow(np.hstack(tuple(roiVisList)))
            plt.figure("Detecting text regions")
            plt.imshow(np.hstack(tuple(textVisList)))
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
