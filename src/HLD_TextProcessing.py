import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math
import HLD_Misc as imgmisc
import HLD_Transform as transform

def find_MSER(imgray,minArea,maxArea,delta):

    mser = cv2.MSER_create()
    mser.setMaxArea(maxArea)
    mser.setMinArea(minArea)
    mser.setDelta(delta)

    regions, _ = mser.detectRegions(imgray)

    hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions]

    areaFilter = []
    for i,region in enumerate(hulls):
        area = cv2.contourArea(region)
        if area >= minArea and area <= maxArea:
            areaFilter.append(regions[i])

    vis = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(vis,[cv2.convexHull(p.reshape(-1,1,2)) for p in areaFilter],-1,255,2)

    return areaFilter,vis

def space_out_text(textImg,textRegions):

    #sort textRegions by x position
    textRegions.sort(key = lambda r: cv2.boundingRect(r)[0])

    minX    = cv2.boundingRect(textRegions[0])[0]
    maxX    = cv2.boundingRect(textRegions[len(textRegions)-1])[0]

    #calculate the average width of textRegions
    avgWidth = sum([cv2.boundingRect(r)[3] for r in textRegions])/len(textRegions)
    outImg = np.zeros((500,int(((maxX - minX) + avgWidth*len(textRegions)) * 2)))
    wordMask = np.zeros(textImg.shape,np.uint8)
    cv2.drawContours(wordMask,textRegions,0,255,-1)
    textColor = detect_text_color(textImg,wordMask)

    #space each region by the average width of the letters
    for i,r in enumerate(textRegions):
        mask = np.zeros(textImg.shape,np.uint8)
        cv2.drawContours(mask,[r],0,255,-1)
        if textColor == 'white':
            imgText = cv2.bitwise_and(textImg,textImg,mask=mask)
        elif textColor == 'black':
            imgText = cv2.bitwise_and(255 - textImg,255 - textImg,mask=mask)
        #Translate the image
        outImg = outImg + transform.translate(imgText,i*avgWidth*0.3,0,outImg.shape)

    #translate final image back
    outImg = transform.translate(outImg,-minX + 10,0,outImg.shape)

    return outImg

def detect_text_color(textImg,mask):
    whiteText = cv2.bitwise_and(textImg,textImg,mask=mask)
    blackText = cv2.bitwise_and(255 - textImg,255 - textImg,mask=mask)
    #morph close each to better detect
    wclosing = cv2.erode(whiteText,np.ones((7,7),np.uint8),iterations = 1)
    bclosing = cv2.erode(blackText,np.ones((7,7),np.uint8),iterations = 1)

    if(np.bincount(wclosing.flatten(),minlength=2)[-1] > np.bincount(bclosing.flatten(),minlength=2)[-1]):
        imgText = 'white'
    else:
        imgText = 'black'

    return imgText

def perform_adaptive_thresh(imgBGR,threshBlock,threshC):
    imgHSV = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(imgHSV)
    #Adaptive gaussian on v channel
    vThresh = cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,
                                    threshBlock,threshC)

    return vThresh

#This function attemps to cluster regions based ellipse, the ratio of semi major axis to minor axis
def filter_regions_by_eccentricity(regions,maxEccentricity):

    filtered = []

    for r in regions:
        (x,y),(MA,ma),_ = cv2.fitEllipse(r)
        (_,_),(_,_),angle = cv2.minAreaRect(r)
        eccentricty = (1 - MA/ma)**(0.5)

        if eccentricty < maxEccentricity and (abs(angle - 45) > 15 != abs(angle - (-45)) > 15):
            filtered.append(r)

    return filtered

def filter_regions_by_location(regions,rect):
    x1,y1,w1,h1 = rect
    filtered = []
    for r in regions:
        (x,y),_ = cv2.minEnclosingCircle(r)
        if (x >= x1 and x <= (x1 + w1)) and (y >= y1 and y <= (y1 + h1)):
            filtered.append(r)

    return filtered

def filter_regions_by_yCluster(regions,minY,maxY,res):

    numBins = int((maxY - minY)/res)
    clusters = [[] for i in range(numBins)]
    histArr = []

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        if y >= minY and y <= maxY:
            histArr.append(y)

    hist,binEdge = np.histogram(histArr,bins=numBins,range=(minY,maxY))
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)

        histIdx = np.abs(binEdge - y).argmin()
        if y < binEdge[histIdx] or histIdx == numBins:
            histIdx = histIdx - 1

        if hist[histIdx] >= 3:
            clusters[histIdx].append(r)

    return [c for c in clusters if len(c) > 0 ]

    return filtered

def filter_regions_by_textHomogeneity(regions,minHeight,maxHeight,res):

    numBins = int((maxHeight - minHeight)/res)
    clusters = [[] for i in range(numBins)]
    histArr = []

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        if h >= minHeight and h <= maxHeight:
            histArr.append(h)

    hist,binEdge = np.histogram(histArr,bins=numBins,range=(minHeight,maxHeight))
    for r in regions:
        x,y,w,h = cv2.boundingRect(r)

        histIdx = np.abs(binEdge - h).argmin()
        if h < binEdge[histIdx]:
            histIdx = histIdx - 1

        if hist[histIdx] >= 3:
            clusters[histIdx].append(r)

    return [c for c in clusters if len(c) > 0 ]


def filter_overlapping_regions(regions):

    filtered = []

    #sort the contours by area - largest to smallest
    regions.sort(key=lambda r: cv2.contourArea(r),reverse=True)

    for r1 in regions:
        (x,y),_,_ = cv2.minAreaRect(r1)
        overlapping = False

        for r2 in filtered:
            leftmost = r2[r2[:,0].argmin()]
            rightmost = r2[r2[:,0].argmax()]
            topmost = r2[r2[:,1].argmin()]
            bottommost = r2[r2[:,1].argmax()]

            if int(x) > leftmost[0] and int(x) < rightmost[0] and int(y) > topmost[1] and y < bottommost[1]:
                overlapping = True
                break

        if not overlapping:
            filtered.append(r1)

    return filtered
