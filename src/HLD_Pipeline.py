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

import HLD_Helper as imghelp
import HLD_TextProcessing as textproc
import HLD_RegionsProc as regionproc
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

def find_text(textImg,config='-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
    cv2.imwrite("test.png",255 - textImg)

    return pytesseract.image_to_string(Image.open('test.png'),config=config)

def extract_hazard_label_text(roiBGR,tuner):

    text = ""
    textVis = []

    vThresh    = imgmisc.perform_adaptive_thresh(roiBGR,tuner.threshBlock,tuner.threshC)
    mserRegion = regionproc.find_MSER(vThresh,tuner.minBlobArea,tuner.maxBlobArea,tuner.blobDelta)
    filtered   = regionproc.filter_regions_by_eccentricity(mserRegion,tuner.maxE)
    filtered   = regionproc.filter_overlapping_regions(filtered)

    regionCluster = regionproc.approx_homogenous_regions_chain(filtered,3,0.2,0.2)
    #sort left to right
    regionCluster = regionproc.sort_left_right(regionCluster)
    for regions in regionCluster:
        space = sum([cv2.boundingRect(r)[2] for r in regions])/len(regions)
        textImg = textproc.space_out_text(roiBGR,regions,space)
        textTmp = find_text(textImg)
        text = text + textTmp
        textVis.append((textImg,textTmp))
        text = text + '\n'





    #for visual stuff
    vis = cv2.cvtColor(vThresh,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(vis,[cv2.convexHull(r) for r in mserRegion],-1,(0,255,0),1)
    cv2.drawContours(vis,[cv2.convexHull(r) for r in filtered],-1,(0,0,255),1)
    cv2.drawContours(vis,[cv2.convexHull(r) for r in filtered],-1,(255,0,0),1)
    for img,txt in textVis:
        plt.figure(txt)
        plt.imshow(img,cmap='gray')

    return ' '.join(text.split()),vis

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
        label,textVis = extract_hazard_label_text(imgROI,tuner)
        #approxlabel = textproc.approximate_label(label,"dictionary.txt")
        print(label)
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
