#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import HLD_Helper as imghelp
import pytesseract
import sys
from PIL import Image
import Transform


def find_region_of_interest(imgray):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    hazardlabels_contours_mask = []
    res,contours,hierachy = imghelp.find_contours(imgray,mask=None)
    rects = imghelp.filter_rectangles(contours)

    #get the largest rectangle
    black = np.zeros(imgray.shape,np.uint8)
    displayMask = []
    for rectContour in rects:
        mask = np.zeros(imgray.shape,np.uint8)
        cv2.fillPoly(mask,[rectContour],255)
        hazardlabels_contours_mask.append((rectContour,mask))
        displayMask.append(mask)

    return hazardlabels_contours_mask

#Imports a region of interest and
def identify_text(imgBGR,mask=None,display=False):

    #First lets apply some thresholding to get the text area
    #regionsMSER = imghelp.find_MSER(imgBGR,mask,display)
    #apply some thresholding
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
    imgray = imgray | (255 - mask)
    thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)[1]

    #lets just look crop the middle third of the image
    x,y,w,h = cv2.boundingRect(mask)

    thresh = thresh[y+int(h/3):y+int(2*h/3),:]

    #Lets try resizing the image
    cv2.imwrite("tmp.png",thresh)
    text = pytesseract.image_to_string(Image.open("tmp.png"))
    print(text)

def white_balancing(imgBGR,mask=None):
    imgYCrCb = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2YCrCb)
    y,Cr,Cb = cv2.split(imgYCrCb)

    #Perform histogram equalization on the y channel
    yEqu = cv2.equalizeHist(y)
    imgYCrCb = cv2.merge((yEqu,Cr,Cb))

    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
    imgrayequ = cv2.equalizeHist(imgray)

    return cv2.cvtColor(imgrayequ,cv2.COLOR_GRAY2BGR)



#The main pipe line
def main():
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        hazardlabels = []

        imgpath = sys.argv[1]

        imgBGR = cv2.imread(imgpath)

        imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
        hl_c_m = find_region_of_interest(imgray)

        ROIList = []
        for i, (rectContour,mask) in enumerate(hl_c_m):
            imgROI = Transform.perspective_trapezoid_to_rect(imgBGR,rectContour,mask)
            ROIList.append(cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB))

        plt.figure("Hazard Label Detection")
        plt.subplot(211)
        plt.imshow(cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB))
        if len(ROIList) > 0:
            plt.subplot(212)
            plt.imshow(np.hstack(tuple(ROIList)))
        else:
            print("No ROI found")

        plt.show()


if __name__ == '__main__':
    main()
