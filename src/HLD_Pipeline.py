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

MINAREA = 100


def find_region_of_interest(imgray,display=False):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    hazardlabels_contours_mask = []

    res,contours,hierachy = imghelp.find_contours(imgray,mask=None)
    rects = imghelp.filter_rectangles(contours)


    #get the largest rectangle
    black = imgray & 0
    displayMask = []
    for rectContour in rects:
        mask = imgray & 0
        rect = cv2.minAreaRect(rectContour) #rect = center(x,y),(width,height),angle
        area = rect[1][0] * rect[1][1]
        if area > imgray.shape[0] * imgray.shape[1] * 0.04:
            cv2.fillPoly(mask,[rectContour],255)
            hazardlabels_contours_mask.append((rectContour,mask))
            rect = cv2.minAreaRect(rectContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            displayMask.append(mask)


    if display:
        vis = imgray.copy() & 0
        cv2.drawContours(vis,contours,-1,255)

        vismask = black.copy()
        for mask in displayMask:
            vismask = mask + vismask

        roi = cv2.bitwise_and(imgray,imgray,mask=vismask)
        plt.figure("ROI")
        plt.imshow(np.hstack((vis,roi)),cmap='gray')

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



def main():
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        hazardlabels = []

        imgpath = sys.argv[1]

        imgBGR = cv2.imread(imgpath)
        imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
        hl_c_m = find_region_of_interest(imgray,display=True)

        """for i, v in enumerate(hl_c_m):
            recContour,mask = v
            print('label ' + str(i))
            for k,v in imghelp.calculate_color_percentage(imgBGR,mask=mask).items():
                print(k + ': ' + str(v[0]),end=', ')
            print()"""

            #imgROI = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)
            #hl = Transform.perspective_trapezoid_to_rect(imgROI,recContour,mask)
        #identify_text(imgBGR,mask=hlMask[0],display=True)

        plt.show()


if __name__ == '__main__':
    main()
