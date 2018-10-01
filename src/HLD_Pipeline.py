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
import ShapeContext

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
    contours = shapeproc.filter_contour_area(contours,minROIArea,maxROIArea)
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

def find_text(textImg,config='-l eng -oem 3 -psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
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
    classRegionsCluster = regionproc.approx_homogenous_regions_chain(classRegions,3,3,3,minLength=1)
    for regions in classRegionsCluster:
        filtered = regionproc.filter_regions_by_area(regions,150,6000)
        if len(filtered) > 0:
            classImg = textproc.space_out_text(roiBGR,filtered,10)
            classImg = transform.translate(classImg,0,-250,classImg.shape)
            tmp = find_text(classImg,config='-l eng --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.')
            matches = re.findall(r"[0-9]|\.(?=[0-9])",tmp)
            classNumber = classNumber + ''.join(matches)

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

def detect_color(imgROI,mask=None):
    topMap = colorproc.calculate_color_percentage(imgROI[0:250,...],mask[0:250,...])
    botMap = colorproc.calculate_color_percentage(imgROI[250:499,...],mask[250:499,...])
    topColor = list(topMap.keys())[0]
    botColor = list(botMap.keys())[0]

    return topMap,botMap

def find_matches(img1,img2):
    #first appply adaptive thresh and morph to remove noise and shadows
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3),plt.show()
    return len(matches)

def find_symbol_cnt(imgROI):
    gauss = cv2.GaussianBlur(imgROI,(5,5),0)
    gray = cv2.cvtColor(gauss,cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,3)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),dtype=np.uint8))
    cnts = cv2.findContours(opening,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    filtered = regionproc.filter_regions_by_location(cnts,(150,50,200,200))
    filtered = regionproc.filter_regions_by_area(filtered,20,9500)

    return filtered

#find the contours of this image
#The main pipe line
def run_detection(imgpath,display):
    tuner = HLD_Tuner.Tuner()
    sc    = ShapeContext.ShapeContext()

    if display:
        textVisList = []
        roiVisList = []
        symbolVisList = []

    imgBGR = cv2.imread(imgpath)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(imgray,tuner.medianKSize)
    blurred = cv2.GaussianBlur(median,tuner.gaussKSize,tuner.gaussSigmaX)
    hl_c_m = find_region_of_interest(blurred,tuner)
    symbolsDict = {
                    'Flame'       : np.load("res/ShapeDescriptors/FlameSymbol.npy"),
                    'Corrosive'   : np.load("res/ShapeDescriptors/CorrosiveSymbol.npy"),
                    'Radioactive' : np.load("res/ShapeDescriptors/RadioactiveSymbol.npy"),
                    'Toxic'       : np.load("res/ShapeDescriptors/ToxicSymbol.npy"),
                    'Oxidizer'    : np.load("res/ShapeDescriptors/OxidizerSymbol.npy"),
                    'Explosive'   : np.load("res/ShapeDescriptors/ExplosiveSymbol.npy")
                  }

    for i, (rectContour,mask) in enumerate(hl_c_m):
        roiMask = 255 - np.zeros(imgBGR.shape[:-1],dtype=np.uint8)
        imgROI = transform.perspective_trapezoid_to_rect(imgBGR,rectContour,tuner.finalSize,mask)
        roiMask = transform.perspective_trapezoid_to_rect(roiMask,rectContour,tuner.finalSize,mask)

        (label,classNo),textVis,nonRegThresh = extract_hazard_label_text(imgROI,tuner)
        topColors,botColors = detect_color(imgROI,mask=roiMask - (255 - nonRegThresh))
        symbolCnts = find_symbol_cnt(imgROI)
        symbolPts  = sc.get_points(symbolCnts)
        symbolDes  = sc.compute_shape_descriptor(symbolPts)

        matches = np.array([sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Flame'],symbolDes)),
                            sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Corrosive'],symbolDes)),
                            sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Radioactive'],symbolDes)),
                            sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Toxic'],symbolDes)),
                            sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Oxidizer'],symbolDes)),
                            sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolsDict['Explosive'],symbolDes))
                            ])

        symbol = list(symbolsDict.keys())[matches.argmin()]

        #symbolPoints = sc.get_points(imgROI[0:210,:])
        #symbolSC     = sc.compute_shape_descriptor(symbolPoints)
        #print(sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolSC,descriptors['Corrosive'])))
        #print(sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolSC,descriptors['Flame'])))
        #print(sc.compute_min_cost_greedy(sc.calc_cost_matrix(symbolSC,descriptors['Radioactive'])))
        print("TOP         : " + list(topColors.keys())[0])
        print("BOTTOM      : " + list(botColors.keys())[0])
        print("LABEL       : " + label)
        print("CLASS NUMBER: " + classNo)
        print("SYMBOL      : " + symbol)

        if display:
            tmpVis = np.zeros(imgROI.shape[:2],dtype=np.uint8)
            cv2.drawContours(tmpVis,symbolCnts,-1,255,1)
            roiVisList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))
            textVisList.append(textVis)
            symbolVisList.append(tmpVis)

    #now sanity check the text list
    if display:
        plt.figure("Input Image")
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(roiVisList) > 0:
            plt.figure("Detecting hazard label")
            plt.imshow(np.hstack(tuple(roiVisList)))
            plt.figure("Detecting text regions")
            plt.imshow(np.hstack(tuple(textVisList)))
            plt.figure("Detecting symbols")
            plt.imshow(np.hstack(tuple(symbolVisList)),cmap='gray')
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
