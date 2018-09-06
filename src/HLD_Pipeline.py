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

MINAREA = 100

def find_region_of_interest(imgray,display=False):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    hazardlabels_mask = []

    res,contours,hierachy = imghelp.find_contours(imgray,mask=None)
    rects = imghelp.filter_rectangles(contours)


    #get the largest rectangle

    black = imgray & 0

    for rectContour in rects:
        rect = cv2.minAreaRect(rectContour) #rect = center(x,y),(width,height),angle
        area = rect[1][0] * rect[1][1]
        if area > imgray.shape[0] * imgray.shape[1] * 0.04:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mask = cv2.drawContours(black,[box],0,255,thickness=-1)
            hazardlabels_mask.append(mask)


    if display:
        vis = imgray.copy() & 0
        cv2.drawContours(vis,contours,-1,255)

        vismask = black.copy()
        for mask in hazardlabels_mask:
            vismask = mask + vismask

        roi = cv2.bitwise_and(imgray,imgray,mask=vismask)
        plt.figure("ROI")
        plt.imshow(np.hstack((vis,roi)),cmap='gray')

    return hazardlabels_mask

#Imports a region of interest and
def identify_text(imgBGR,mask=None,display=False):

    #First lets apply some thresholding to get the text area
    #regionsMSER = imghelp.find_MSER(imgBGR,mask,display)
    #apply some thresholding
    imgBGR = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)
    imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
    imgray = cv2.resize(imgray,None,fx=1/4,fy=1/4,interpolation=cv2.INTER_AREA)

    #Lets try resizing the image
    cv2.imwrite("tmp.png",imgray)
    text = pytesseract.image_to_string(Image.open("tmp.png"))
    print(text)



def main():
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        imgpath = sys.argv[1]

        imgBGR = cv2.imread(imgpath)
        imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
        hlMask = find_region_of_interest(imgray,display=True)
        identify_text(imgBGR,mask=hlMask[0],display=True)

        plt.show()


if __name__ == '__main__':
    main()
