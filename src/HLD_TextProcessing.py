import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math
import re

import HLD_Misc as imgmisc
import HLD_RegionsProc as regionproc
import HLD_Transform as transform

def approximate_label(text,wordFile):

    outText = ""
    match   = []
    lines = text.split('\n')
    words = [''.join(re.findall(r"[a-zA-Z]",l)) for l in lines]

    f = open(wordFile)
    data = f.read()
    expectedWords = data.split('\n')
    for word in words:
        lcs = set()
        word = word.upper()
        noMatch = False
        while len(word) >= 3 and noMatch == False:
            matches = []
            for expectword in expectedWords:
                lcs = find_LCS(expectword,word)
                if len(lcs) > 0:
                    substring  = list(lcs).pop()
                    percentMatch = 0 if len(expectword) == 0 else len(substring)/len(expectword)
                    if percentMatch == 1:
                        matches.append((substring,expectword))

            if len(matches) > 0:
                matches.sort(key=lambda x: len(x[1]),reverse = True)
                longestSubstring, closestMatch = matches[0]
                outText = outText + closestMatch + ' '
                word = word.replace(longestSubstring,'')
            else:
                noMatch = True

        if outText == "":
            outText = '\n'.join([word for word in words if len(word) >= 3])

    return outText

def space_out_text(imgBGR,textRegions,spaceWidth):

    #print(len(textRegions))
    textRegions.sort(key = lambda r: cv2.boundingRect(r)[0])
    outImg = np.zeros((imgBGR.shape[0],int(imgBGR.shape[1] + spaceWidth*len(textRegions))))
    currX = cv2.boundingRect(textRegions[0])[0] + 20
    gaussThresh = remove_Gaussian_noise(imgBGR,textRegions)
    for i,r in enumerate(textRegions):
        textColor = detect_text_color(gaussThresh,r)
        mask = imgmisc.get_mask_from_regions([r],gaussThresh.shape)
        if textColor == 'black': gaussThresh = 255 - gaussThresh
        regionImg = cv2.bitwise_and(gaussThresh,gaussThresh,mask=mask)

        x,_,width,_ = cv2.boundingRect(r)
        outImg = outImg + transform.translate(regionImg,currX - x,0,outImg.shape)
        currX = currX + width + spaceWidth

    return outImg

def remove_Gaussian_noise(imgBGR,regions):
    blurSize = int(math.sqrt(sum([cv2.contourArea(r) for r in regions])/len(regions))/1.7) * 2 + 1
    blur   = cv2.GaussianBlur(imgBGR,(blurSize,blurSize),0)
    thresh = imgmisc.perform_adaptive_thresh(blur,25,2)

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

#This function attemps to cluster regions based ellipse, the ratio of semi major axis to minor axis

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
