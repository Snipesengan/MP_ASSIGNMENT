import cv2
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import math
import sys

import HLD_ShapeProc as shapeproc
import HLD_Misc as imgmisc

class ShapeContext(object):

    def __init__(self,nBinsR=5,nBinsTheta=12,rInner=0.1250,rOuter=2):
        self.nBinsR        = nBinsR
        self.nBinsTheta    = nBinsTheta
        self.rInner        = rInner
        self.rOuter        = rOuter
        self.nBins         = nBinsTheta*nBinsR

    #Compute the eucledian distance between points
    #Where result[i,j] = eucledianDist(P[i],Q[j])
    def _calc_dist_arr(self,P,Q):
        result = np.zeros((len(P),len(Q)))
        for i in range(len(P)):
            for j in range(len(Q)):
                v = np.array(P[i]) - np.array(Q[j])
                result[i,j] = math.sqrt(v[0]**2 + v[1]**2)


        return result

    def _calc_ang_arr(self,P,Q):
        result = np.zeros((len(P),len(Q)))
        for i in range(len(P)):
            for j in range(len(Q)):
                v = np.array(P[i]) - np.array(Q[j])
                result[i,j] = math.atan2(v[1],v[0])

        return result

    def get_points(self,cnts,sampleto=130):
        pts = np.array(cnts[0].reshape(-1,2))
        for i in range(1,len(cnts)):
            pts = np.concatenate([pts,np.array(cnts[i].reshape(-1,2))])

        sampleIdxs = np.random.choice(pts.shape[0],sampleto)
        sampPts = pts[sampleIdxs]

        return sampPts

    #Cost between two points descriptor
    def _cost(self,g,h):
        cost = 0
        cost = np.sum(np.divide((g - h)**2,(g + h),where=(g+h) != 0))
        #cost = np.sum((gCopy - hCopy)**2/(gCopy + hCopy))
        """
        for k in range(self.nBinsTheta * self.nBinsR):
            if g[k] + h[k]:
                cost += ((g[k] - h[k])**2)/(g[k] + h[k])
        """

        return 0.5 * cost

    #Compute cost matrix between two shape descriptor P and Q
    def calc_cost_matrix(self,P,Q,qlength=None):
        p,_ = P.shape
        p2,_ = Q.shape
        d = p2

        if qlength:
            d = qlength

        C = np.zeros((p,p2))
        for i in range(p):
            for j in range(p2):
                C[i,j] = self._cost(Q[j] / d, P[i] / p)

        return C

    def compute_shape_descriptor(self,points):

        numPts = len(points)

        #Get the relative distance array where arr[i,j] is relative dist from Pi to Pj
        rArr = self._calc_dist_arr(points,points)
        #Normalizing the distance will allow some scale invariancy
        rArrNorm = rArr / rArr.mean()

        #Log space for the shape descriptor histogram
        rBinEdges = np.logspace(np.log10(self.rInner),np.log10(self.rOuter),self.nBinsR)
        rArrQ  = np.zeros((numPts,numPts),dtype=int)
        #Summing occurances between interval, Quantization (Cummulative histogram basically)
        for i in range(self.nBinsR):
            rArrQ += (rArrNorm < rBinEdges[i])

        fz = rArrQ > 0

        #Calculating angle between each points
        thetaArr = self._calc_ang_arr(points,points)

        #Shift angle by 2Pi to get angles[0,2pi]
        thetaArr2 = thetaArr + 2 * math.pi * (thetaArr < 0 )
        thetaArrQ = (1 + np.floor(thetaArr2 / (2 * math.pi / self.nBinsTheta))).astype(int)

        #building the descriptor based on angle and distance
        nBins = self.nBinsTheta * self.nBinsR
        descriptor = np.zeros((numPts,nBins))
        for i in range(numPts):
            sn = np.zeros((self.nBinsR,self.nBinsTheta))
            for j in range(numPts):
                if fz[i,j]:
                    sn[rArrQ[i,j] - 1, thetaArrQ[i,j] - 1] += 1
            descriptor[i] = sn.reshape(nBins)

        return descriptor

    def compute_min_cost_greedy(self,C):
        #Greedy choose the higher dimension side for better results
        if C.shape[1] > C.shape[0]:
            C = C.T

        totalCost = 0
        row,col = C.shape
        for i in range(row):
            minIdx = C.argmin()
            totalCost = totalCost + C[int(minIdx/row),minIdx%row]
            #Remove potential of going to this particular point again
            C[:,minIdx%row] = math.inf

        return totalCost

    def diff(self,des1,des2):
        C = self.calc_cost_matrix(des1,des2)
        totalCost = self.compute_min_cost_greedy(C)

        return totalCost

if __name__ == "__main__":
    imgpath = sys.argv[1]
    #Use this to generate Binary array file for shape descriptor
    sc = ShapeContext()
    img  = cv2.imread(sys.argv[1],0)
    #gauss = cv2.GaussianBlur(img,(5,5),0)
    #thresh = imgmisc.perform_adaptive_thresh(gauss,35,2)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3),dtype=np.uint8))
    canny = cv2.Canny(img,100,200)
    cnts = cv2.findContours(canny,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1]
    pts  = sc.get_points(cnts)
    des  = sc.compute_shape_descriptor(pts)


    outputDest = imgpath.replace('res/Symbols','res/ShapeDescriptors').split('.')[0] + '.npy'
    np.save(outputDest,des)
