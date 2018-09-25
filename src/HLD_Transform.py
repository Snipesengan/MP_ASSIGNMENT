import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

def perspective_trapezoid_to_rect(imgBGR,rectContour,finalSize,mask=None):

    imgROI = cv2.bitwise_and(imgBGR,imgBGR,mask=mask)

    leftmost = tuple(rectContour[rectContour[:,:,0].argmin()][0])
    rightmost = tuple(rectContour[rectContour[:,:,0].argmax()][0])
    topmost = tuple(rectContour[rectContour[:,:,1].argmin()][0])
    bottommost = tuple(rectContour[rectContour[:,:,1].argmax()][0])

    fromPoints = np.float32([[bottommost[0],bottommost[1]],
                           [leftmost[0],leftmost[1]],
                           [topmost[0],topmost[1]],
                           [rightmost[0],rightmost[1]]])

    toPoints = np.float32([[finalSize[0]/2,finalSize[1]],
                           [0,finalSize[1]/2],
                           [finalSize[0]/2,0],
                           [finalSize[0],finalSize[1]/2]])

    M = cv2.getPerspectiveTransform(fromPoints,toPoints)
    des = cv2.warpPerspective(imgROI,M,finalSize)

    return des

def translate(img,dx,dy,shape):
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(shape[1],shape[0]))

    return dst
