# Author      : Nhan Dao
# Purpose     : Part of stage 1 of Assignment, identifying background on a
#               computer generated label
# Description : This class should be able to take in an image and report
#               the top and bottom halves of the background

import cv2
import sys
import tkinter
import numpy as np
import matplotlib.pyplot as plt

class ColorDetector:

    #Create a HSV boundary for different color
    NUM_COLOR     = 5
    thresh_dict   = [
                     ('blue',(np.array([98,109,20]),np.array([112,255,255]))),
                     ('yellow',(np.array([20,190,20]),np.array([30,255,255]))),
                     ('orange',(np.array([5,50,50]),np.array([15,255,255]))),
                     ('red',(np.array([0,70,50]),np.array([10,255,255]))),
                     ('red',(np.array([170,70,50]),np.array([180,255,255])))
                    ]

    def __init__(self):
        pass

    def detect(self, img):
        count = np.zeros((256))
        for color in self.thresh_dict:

            #Iterate throught the thresh hold dict
            mask = cv2.inRange(img,color[1][0],color[1][1])
            res = cv2.bitwise_and(img,img,mask=mask)

            #Count how many colors there are
            count = np.vstack((np.bincount(mask.flatten(),minlength=256),count))

        return (self.thresh_dict[self.NUM_COLOR - np.argmax(count[:,255]) - 1])[0]

#Test harness
if __name__ == '__main__':

    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        cd  = ColorDetector()
        hsv = cv2.cvtColor(cv2.imread(sys.argv[1]),cv2.COLOR_BGR2HSV)
        h   = hsv.shape[1]
        mid = int(h/2)

        print("Top: " + cd.detect(hsv[:mid,:,:]))
        print("Bottom: " + cd.detect(hsv[mid+1:,:,:]))
