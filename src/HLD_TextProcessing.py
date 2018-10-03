import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math
import re

import HLD_Misc as imgmisc
import HLD_RegionsProc as regionproc
import HLD_Transform as transform

def space_out_text(imgBGR,textRegions,spaceWidth):

    #Sorts the x coordinate, positioning the regions from left to right
    textRegions.sort(key = lambda r: cv2.boundingRect(r)[0])

    #Create an output image, the final image should have enough width to have all the regions
    #spaced out
    outImg = np.zeros((imgBGR.shape[0],int(imgBGR.shape[1] + spaceWidth*len(textRegions))))
    currX = cv2.boundingRect(textRegions[0])[0] + 20

    #Remove the gaussian noise on the image, dynamically calculate size of window to blur based
    #on the area of the textRegions (See function).
    gaussThresh = remove_Gaussian_noise(imgBGR,textRegions)

    #Go through each regions and space them out, a bit of geometry
    for i,r in enumerate(textRegions):
        #Find the text color of region, this is important because we want to remove more noise
        #based on this (dilate/erode) and also correctly binarize to have black text on white
        textColor = detect_text_color(gaussThresh,r)
        if textColor == 'black':
            gaussThresh = 255 - gaussThresh
        dilate = cv2.erode(gaussThresh,np.ones((1,1),np.uint8),iterations = 1)

        #Apply the mask using the region and translate the region to the appropriate location
        mask = imgmisc.get_mask_from_regions([r],gaussThresh.shape)
        regionImg = cv2.bitwise_and(dilate,dilate,mask=mask)
        x,_,width,_ = cv2.boundingRect(r)
        outImg = outImg + transform.translate(regionImg,currX - x,0,outImg.shape)
        currX = currX + width + spaceWidth

    return outImg

def remove_Gaussian_noise(imgBGR,regions):
    blurSize = int(math.sqrt(sum([cv2.contourArea(r) for r in regions])/len(regions))/1.7) * 2 + 1
    blur   = cv2.GaussianBlur(imgBGR,(blurSize,blurSize),0)
    median = cv2.medianBlur(blur,3)
    thresh = imgmisc.perform_adaptive_thresh(median,35,2,np.ones((0,0),np.uint8))


    return thresh

def detect_text_color(textImg,textRegion):
    wordMask = np.zeros(textImg.shape,np.uint8)
    cv2.drawContours(wordMask,[textRegion],0,255,-1)
    whiteText = cv2.bitwise_and(textImg,textImg,mask=wordMask)
    blackText = cv2.bitwise_and(255 - textImg,255 - textImg,mask=wordMask)
    if(np.bincount(whiteText.flatten(),minlength=2)[-1] > np.bincount(blackText.flatten(),minlength=2)[-1]):
        textColor = 'white'
    else:
        textColor = 'black'

    return textColor

def find_LCS(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set
