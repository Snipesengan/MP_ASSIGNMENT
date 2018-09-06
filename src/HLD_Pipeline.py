#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import HLD_Helper as imghelp
import sys

MEDIAN_BLUR_KSIZE = 21
CUTOFF_VAL = 100

def find_region_of_interest(imgray,display=False):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label *Knowledge engineering here*

    hazardlabels_mask = []

    res,contours,hierachy = imghelp.find_contours(imgray,mask=None)
    rects = imghelp.filter_rectangles(contours)


    #get the largest rectangle

    black = imgray & 0

    for rectContour in rects:
        rect = cv2.minAreaRect(rectContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        mask = cv2.drawContours(black,[box],0,255,thickness=-1)
        hazardlabels_mask.append(mask)


    if display:
        vis = imgray.copy() & 0
        cv2.drawContours(vis,contours,-1,255)

        vismask = black.copy()
        for mask in hazardlabels_mask:
            mask = mask + vismask

        roi = cv2.bitwise_and(imgray,imgray,mask=mask)
        plt.figure("ROI")
        plt.imshow(np.hstack((vis,roi)),cmap='gray')

    return hazardlabels_mask

#Imports a region of interest and
def identify_text(roiBGR,display=False):
    



def main():
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:

        imgBGR = cv2.imread(sys.argv[1])
        imgray = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2GRAY)
        hlMask = find_region_of_interest(imgray,display=True)
        imghelp.localize_text_in_image(imgBGR,mask=hlMask[0],display=True)

        plt.show()


if __name__ == '__main__':
    main()
