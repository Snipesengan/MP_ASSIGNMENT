import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt

def perspective_trapezoid_to_rect(imgROI,rectContour,mask):


    rect = cv2.minAreaRect(rectContour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #use hough lines to find the 4 points of the mask
    lines = cv2.HoughLines(mask, 1, np.pi / 180, 150, None, 0, 0)

    """if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)"""

    #find the points in which corners are detected

    #M = cv2.getPerspectiveTransform(corners,box)
    #des = cv2.warpPerspective(imgROI,M,(500,500))
