#HDL stands for Hazmat Label Detector :P
#This pipeline will identify and classify Hazmat Labels :P
#Ultimately I want to feed the data into a SVM classifier - let see how good that turns out

import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt



def find_region_of_interest(img):
    #Find region of interest, essential: look for things that might
    #look like a Hazmat label.

    
