import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import sys
import math

def display_region(regions,dim):
    mask = np.zeros(dim)
    cv2.drawContours(mask,regions,-1,255,1)
    plt.figure("Regions")
    plt.imshow(mask,cmap='gray')
    plt.show()

def calculate_solidity(regions):
    left = []
    right = []
    top = []
    bottom = []
    regionArea = 0

    for r in regions:
        x,y,w,h = cv2.boundingRect(r)
        left.append(x)
        right.append(x+w)
        top.append(y)
        bottom.append(y+h)

        regionArea = regionArea + cv2.contourArea(r)

    x1 = left[np.array(left).argmin()]
    y1 = top[np.array(top).argmin()]
    x2 = right[np.array(right).argmax()]
    y2 = bottom[np.array(bottom).argmax()]

    boxArea = (y2 - y1) * (x2 - x1)

    return (regionArea/boxArea) * 100

def calculate_centroid(regions):

    centroids = []

    #Calculate the centroid
    for r in regions:
        M = cv.moments(r)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centroids.append((cx,cy))

    return centroids
