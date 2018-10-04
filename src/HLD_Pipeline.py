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
import threading
import time

import HLD_ShapeProc as shapeproc
import HLD_TextProcessing as textproc
import HLD_RegionsProc as regionproc
import HLD_Transform as transform
import HLD_Tuner
import HLD_ColorProc as colorproc
import HLD_Misc as imgmisc
import ShapeContext

#Imports: a gray scale image, a tuner object
#Exports: Contours and corresponding mask
#         of areas that resembles a diamond
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

    #Find contours of image
    res,contours,hierachy = shapeproc.find_contours(imgray,cannyMin,cannyMax,morphK)

    #Filter contours based on area, labels are atleast 2000pixels
    contours = shapeproc.filter_contour_area(contours,minROIArea,maxROIArea)

    #Filter out any contours that resembles a rectangle
    rects = shapeproc.filter_rectangles(contours,epsilon)

    #Filter overlapping contours (This is to get rid of contours that is a part
    # or inside a bigger contour)
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

#Import: an image containing text to be read by Tesseract
#Export: The output of Tesseract
def find_text(textImg,config='-l eng --psm 7 -c tessedit_char_whitelist=\
              ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
    #Erode the text to better allow tesseract to detect, this may be
    # a bug/internal operations that requires big text to be eroded
    erode = cv2.erode(textImg,np.ones((2,2),dtype=np.uint8),iterations=1)

    cv2.imwrite("tesscache.png",255 - erode)
    return pytesseract.image_to_string(Image.open('tesscache.png'),config=config)


#Import: roiBGR, tuner object
#Export: Descritopn of the hazard sign (color,label,symbol,classNo)
def extract_hazard_label_text(roiBGR,tuner):
    text = ""
    classNumber = ""
    textVis = []

    #Pre processing, this is to help remove some noise,
    blur       = cv2.medianBlur(roiBGR,3,3)
    #Adaptive threshold to binarize/remove shadows
    vThresh    = imgmisc.perform_adaptive_thresh(blur,tuner.threshBlock,tuner.threshC,tuner.threshErode)
    #Find MSER regions, the function to do this is tuned to be sensitive to blob that are features of
    #hazard signs such as symbol & text.
    mserRegion = regionproc.find_MSER(vThresh,tuner.minBlobArea,tuner.maxBlobArea,tuner.blobDelta)
    #Filter some regions based on their eccentricity, remove regions that are too eccentric, this
    #specifically targets the blob that does not resembles text/symbols because of the blocky nature
    #of these features, their eccentricity will be close to that of a circle
    eccfiltered   = regionproc.filter_regions_by_eccentricity(mserRegion,tuner.maxE)
    filtered   = regionproc.filter_overlapping_regions(eccfiltered)

    #For Text
    #Since specific characteristics of text regions are known (See documentation),
    #a depth first search between regions to cluster regions text
    #Meaning each clusters contain regions (mutiple MSER regions), that are similar
    #in height, vertical and horizontal distance
    #Hence the goal is to find regions that represent words and process them further
    #to feed them into the OCR.
    regionCluster = regionproc.approx_homogenous_regions_chain(filtered,3,0.2,0.2,minLength=1)
    #Sorts region to appear in English reading order (left to right, top to bottom)
    regionCluster = regionproc.sort_left_to_right(regionCluster)
    #Go to each cluster and find the text
    for regions in regionCluster:
        #find the average space between each regions
        space = sum([cv2.boundingRect(r)[2] for r in regions])/len(regions)
        #correct the text; space them out and make black text white
        textImg = textproc.correct_text_regions(roiBGR,regions,space)
        cv2.imwrite("Text.png",textImg)
        tessOut =  find_text(textImg).upper()
        textTmp = ''.join(re.findall(r"[A-Z]|!",tessOut))
        textVis.append(regions)

        #Some basic sanity checking, there is a substring within the text thats the same as
        #any words in the dictionary
        with open("dictionary.txt") as f:
            longest = ""
            dictionary = f.read().split('\n')
            for words in dictionary:
                lcs = textproc.find_LCS(textTmp,words)
                if len(lcs) > 0:
                    tmp = lcs.pop()
                    if len(tmp) > len(longest): longest = tmp
            if len(longest) >= 3:
                #Cases where text appears joined together
                if textTmp == 'FLAMMABLEGAS':
                    textTmp = 'FLAMMABLE GAS'
                elif textTmp == 'NONFLAMMABLEGAS':
                    textTmp = 'NON-FLAMMABLE GAS'
                elif textTmp == 'ORGANICPEROXIDE':
                    textTmp = 'ORGANIC PEROXIDE'

                text = text + textTmp + ' '

    #Strip spaces
    text = re.sub(r" (?= )","",text.strip())

    #FOR CLASS NUMBER
    #get only the bottom regions where the class number is, similar to text but with
    #different filtering parameter
    classRegions = regionproc.filter_regions_by_location(eccfiltered,(150,350,150,150))
    classRegions = regionproc.filter_regions_by_area(classRegions,150,6000)
    classRegionsCluster = regionproc.approx_homogenous_regions_chain(classRegions,3,1.2,3,minLength=1)
    classRegionsCluster.sort(key= lambda x: regionproc.calculate_regions_area(x,roiBGR.shape[:2]),reverse=True)
    if len(classRegionsCluster) > 0:
        regions = classRegionsCluster[0]
        classImg = textproc.correct_text_regions(roiBGR,regions,10)
        classImg = transform.translate(classImg,0,-150,classImg.shape)
        tmp = find_text(classImg,config='--psm 6 -c tessedit_char_whitelist=0123456789.')
        #simple sanity checky using regex
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

    if text == "":
        text = "(None)"
    if classNumber == "":
        classNumber = "(None)"

    return (text,classNumber),vis,nonRegThresh

#Imports an image that resembles a hazard label
#Export the most dominant top & bottom colors
def detect_color(imgROI,mask=None):
    topMap = colorproc.calculate_color_percentage(imgROI[0:250,...],mask[0:250,...])
    botMap = colorproc.calculate_color_percentage(imgROI[250:499,...],mask[250:499,...])
    topColor = topMap[0][0]
    botColor = botMap[0][0]

    return topColor,botColor

#Imports a region of interest (hazard label)
#Exports contours used by the shapecontext module to classify symbols
def find_symbol_cnt(imgROI):
    #Naive crop of the image to get just the top half where symbols are contains
    #This needs to be improve
    crop  = imgROI[:230,:,:]
    gauss = cv2.GaussianBlur(crop,(5,5),0) #gauss blur to remove noise
    gray = cv2.cvtColor(gauss,cv2.COLOR_BGR2GRAY).astype(float)
    gray *= 1.5 #improve contrasts a bit
    gray  = np.clip(gray,0,255)
    canny = cv2.Canny(gray.astype(np.uint8),50,100)
    #Find the contours
    cnts = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    #Filter out contours based on their location and area
    filtered = regionproc.filter_regions_by_location(cnts,(150,50,200,170))
    filtered = regionproc.filter_regions_by_area(filtered,20,9500)

    return filtered


def classify_label(imgBGR,rectContour,mask,roiVisList,textVisList,symbolVisList,display=False):
    tuner = HLD_Tuner.Tuner()
    sc    = ShapeContext.ShapeContext()
    #Load binary files containing symbol shape context descriptors
    symbolsDes = [('FLAME',np.load("res/ShapeDescriptors/FlameSymbol.npy")),
                  ('CORROSIVE',np.load("res/ShapeDescriptors/CorrosiveSymbol.npy")),
                  ('RADIOACTIVE',np.load("res/ShapeDescriptors/RadioactiveSymbol.npy")),
                  ('SKULL & BONES ON BLACK DIAMOND',np.load("res/ShapeDescriptors/ToxicSymbol.npy")),
                  ('OXIDIZER',np.load("res/ShapeDescriptors/OxidizerSymbol.npy")),
                  ('EXPLOSIVE',np.load("res/ShapeDescriptors/ExplosiveSymbol.npy")),
                  ('CANNISTER',np.load("res/ShapeDescriptors/CannisterSymbol.npy")),
                  ('1.5',np.load("res/ShapeDescriptors/1_5.npy")),
                  ('1.6',np.load("res/ShapeDescriptors/1_6.npy"))
                  ]

    #Perspective correct the image
    roiMask = 255 - np.zeros(imgBGR.shape[:-1],dtype=np.uint8)
    roiMask = transform.perspective_trapezoid_to_rect(roiMask,rectContour,tuner.finalSize,mask)
    imgROI = transform.perspective_trapezoid_to_rect(imgBGR,rectContour,tuner.finalSize,mask)
    #Find symbol contour
    symbolCnts = find_symbol_cnt(imgROI)
    #If there is no symbol (sanity; images must have a symbol)
    if len(symbolCnts) > 0:
        #Use the shape context module to calculate the symbol shape descriptor based on its contour
        symbolPts  = sc.get_points(symbolCnts)
        symbolDesc  = sc.compute_shape_descriptor(symbolPts)

        #Iterate through the symbols descriptor list and calculate the matching costs to the target symbol
        costdict = {desc[0]:None for desc in symbolsDes}
        for desc in symbolsDes:
            costdict[desc[0]] = sc.diff(desc[1],symbolDesc)

        #Sorts the cost, find the symbol that matched with the lowest cost, meaning its the likely match
        costs = [(k,v) for k,v in costdict.items()]
        costs.sort(key=lambda x:x[1])
        #Sanity checking; If the cost to match is too high, its not likely to be a symbol but a spurious hit
        if costs[0][1] < tuner.maxSymbolCost:
            symbol = costs[0][0]
            (label,classNo),textVis,nonRegThresh = extract_hazard_label_text(imgROI,tuner)
            topColor,botColor = detect_color(imgROI,mask=roiMask - (255 - nonRegThresh))
            print("Top          : %s"%(topColor))
            print("Bottom       : %s"%(botColor))
            print("Class Number : %s"%(classNo))
            print("Label Text   : %s"%(label))
            print("Symbol       : %s"%(symbol))

            #For displaying
            if display:
                tmpVis = np.zeros(imgROI.shape[:2],dtype=np.uint8)
                if len(symbolCnts) > 0:
                    for pts in symbolPts:
                        tmpVis[pts[1],pts[0]] = 255
                roiVisList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))
                textVisList.append(textVis)
                symbolVisList.append(tmpVis)
    else:
        symbol = "No symbol found"


    return

#find the contours of this image
#The main pipe line
def run_detection(imgpath,display):
    startTime = time.time()
    #For displaying stuff
    roiVisList = []
    textVisList = []
    symbolVisList = []

    #Preprocessing
    tuner = HLD_Tuner.Tuner()
    imgBGR = cv2.imread(imgpath)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(imgray,tuner.medianKSize)
    blurred = cv2.GaussianBlur(median,tuner.gaussKSize,tuner.gaussSigmaX)
    #Finds region of interests
    hl_c_m = find_region_of_interest(blurred,tuner)

    #Go through each label and classify them
    for i, (rectContour,mask) in enumerate(hl_c_m):
        classify_label(imgBGR,rectContour,mask,roiVisList,textVisList,symbolVisList,display)

    elapsed = time.time() - startTime
    print("Finished in %.3fs"%(elapsed))
    #For displaying stuff
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
