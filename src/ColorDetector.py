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
    thresh_dict   = [
                     ('blue',(np.array([98,109,20]),np.array([112,255,255]))),
                     ('green',(np.array([36,0,0]),np.array([70,255,255]))),
                     ('yellow',(np.array([20,190,20]),np.array([30,255,255]))),
                     ('orange',(np.array([5,50,50]),np.array([15,255,255]))),
                     ('red',(np.array([0,70,50]),np.array([10,255,255]))),
                     ('red',(np.array([170,70,50]),np.array([180,255,255]))),
                     ('black',(np.array([0,0,0]),np.array([180,255,0])))
                    ]

    def __init__(self):
        pass

    #Detects the majority background colors
    #Works by applying a mask for different colors in HSV space, and picking
    #the color with the highest frequency
    def detectColor(self, img):
        #Convert img to HSV
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        count = np.zeros((2))
        for color in self.thresh_dict:

            #Iterate throught the thresh hold dict
            mask = cv2.inRange(img,color[1][0],color[1][1])
            binCount = np.bincount(mask.flatten(),minlength=256)
            #Count how many colors there are
            count = np.vstack((binCount[[0,-1]],count))

        count = count[0:-1]

        return self.thresh_dict[len(self.thresh_dict) - np.argmax(count[:,-1]) - 1][0]


    def detectShape(self,img):
        #Ok we wil need to detect the shape of the image,
        #and draw a bounding box around it
        pass

#Test harness
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("Usage -- python {script} <image_path>".format(script=sys.argv[0]))
    else:
        cd  = ColorDetector()
        img = cv2.imread(sys.argv[1])
        h   = img.shape[1]
        mid = int(h/2)

        print("Top: " + cd.detectColor(img[:mid,:,:]))
        print("Bottom: " + cd.detectColor(img[mid+1:,:,:]))

        #plt.figure("Image"),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGBA))
        #plt.show()
