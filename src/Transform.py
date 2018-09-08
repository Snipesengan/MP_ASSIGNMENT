import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math

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

    rect = cv2.minAreaRect(rectContour)
    toPoints = cv2.boxPoints(rect)
    toPoints = np.float32(box)

    M = cv2.getPerspectiveTransform(fromPoints,toPoints)
    des = cv2.warpPerspective(imgROI,M,(imgROI.shape[1],imgROI.shape[0]))

    vis = cv2.warpPerspective(mask,M,(imgROI.shape[1],imgROI.shape[0]))
    vis = cv2.merge((vis,mask,np.zeros(mask.shape,np.uint8)))

    #plt.imshow(np.hstack((cv2.cvtColor(imgROI,cv2.COLOR_BGR2RGB),vis,cv2.cvtColor(des,cv2.COLOR_BGR2RGB))))

    return rect,des

def affine_correction(imgROI,rect):

    #Rotate the picture then scale it so that the aspect ratio is now a square hehe xD
    #then rotate back

    (x,y),(w,h),angle = rect
    aspect = w/h
    center = imgROI.shape[0]/2,imgROI.shape[1]/2

    #Translate the image to the center, this so the picture does not clip when rotated
    M = np.float32([[1,0,center[1] - x],[0,1,center[0]-y]])
    dst = cv2.warpAffine(imgROI,M,(imgROI.shape[1],imgROI.shape[0]))

    #Original points of the corner, this will be used to crop the image
    M = cv2.getRotationMatrix2D((imgROI.shape[1]/2,imgROI.shape[0]/2),angle,1)
    rotated = cv2.warpAffine(imgROI,M,(imgROI.shape[1],imgROI.shape[0]))

    if(aspect > 1):
        resized = cv2.resize(rotated,None,fx=1,fy=aspect,interpolation=cv2.INTER_CUBIC)

    else:
        resized = cv2.resize(rotated,None,fx=1/aspect,fy=1,interpolation=cv2.INTER_CUBIC)

    #now rotate it back
    M = cv2.getRotationMatrix2D((center[1],center[0]),45,1/1.4142)
    dst = cv2.warpAffine(resized,M,None)

    #now lets crop
    #x,y,w,h = cv2.boundingRect(box)
    return dst

def _points_transform(points,M):
    dst = []
    for point in points:
        x,y = point
        resX = M[0][0]*x + M[0][1]*y + M[0][2]
        rexY = M[1][0]*x + M[1][1]*y + M[1][2]
        dst.append([x,y])

    dst = np.array(dst,dtype=int)
    dst.reshape(points.shape)

    return dst
