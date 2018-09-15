import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

DEST_SIZE = (500,500) #size fo the final image after perspective transform

def perspective_trapezoid_to_rect(imgROI,rectContour,mask):


    rect = cv2.minAreaRect(rectContour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cnt = rectContour
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    fromPoints = np.float32([[bottommost[0],bottommost[1]],
                           [leftmost[0],leftmost[1]],
                           [topmost[0],topmost[1]],
                           [rightmost[0],rightmost[1]]])

    toPoints = np.float32([[DEST_SIZE[0]/2,DEST_SIZE[1]],
                           [0,DEST_SIZE[1]/2],
                           [DEST_SIZE[0]/2,0],
                           [DEST_SIZE[0],DEST_SIZE[1]/2]])

    M = cv2.getPerspectiveTransform(fromPoints,toPoints)
    des = cv2.warpPerspective(imgROI,M,(500,500))

    return des
