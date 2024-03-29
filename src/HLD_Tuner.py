#This files contains thresholds to tune the HLD_Helper
import cv2
import numpy as np

class Tuner:
    """
        Hazard labels detector tuner, this contains values that can be tuned to better detect
    """

    def __init__(self):
        #pre-processing
        self._medianKSize = 3
        self._gaussKSize  = (5,5)
        self._gaussSigmaX = 0

        #Region of Interest
        self._cannyMax    = 150
        self._cannyMin    = 50
        self._morphK      = np.ones((7,7),np.uint8)
        self._minROIArea  = 20000
        self._maxROIArea  = None
        self._epsilon     = 0.11

        #Geometric Transform
        self._finalSize   = (500,500)

        #Text localization
        self._minBlobArea   = 55
        self._maxBlobArea   = 5000
        self._blobDelta     = 250
        self._threshBlock   = 35
        self._threshC       = 3
        self._threshErode   = np.ones((2,2),np.uint8)
        self._maxE          = 0.95
        
        #Symbol
        self._maxSymbolCost = 45


    #Lets make setter and getters for these things lmao, python sucks at OO
    def medianKSize():
        doc = "The_medianKSize property."
        def fget(self):
            return self._medianKSize
        def fset(self, value):
            self._medianKSize = value
        def fdel(self):
            del self._medianKSize
        return locals()
    medianKSize = property(**medianKSize())

    def gaussKSize():
        doc = "The_gaussK property."
        def fget(self):
            return self._gaussKSize
        def fset(self, value):
            self._gaussKSize = value
        def fdel(self):
            del self._gaussKSize
        return locals()
    gaussKSize = property(**gaussKSize())

    def gaussSigmaX():
        doc = "The gaussSigmaX property."
        def fget(self):
            return self._gaussSigmaX
        def fset(self, value):
            self._gaussSigmaX = value
        def fdel(self):
            del self._gaussSigmaX
        return locals()
    gaussSigmaX = property(**gaussSigmaX())

    def cannyMax():
        doc = "The cannyMax property."
        def fget(self):
            return self._cannyMax
        def fset(self, value):
            self._cannyMax = value
        def fdel(self):
            del self._cannyMax
        return locals()
    cannyMax = property(**cannyMax())

    def cannyMin():
        doc = "ThecannyMin property."
        def fget(self):
            return self._cannyMin
        def fset(self, value):
            self.cannyMin = value
        def fdel(self):
            del self._cannyMin
        return locals()
    cannyMin = property(**cannyMin())

    def morphK():
        doc = "ThemorphK property."
        def fget(self):
            return self._morphK
        def fset(self, value):
            self._morphK = value
        def fdel(self):
            del self._morphK
        return locals()
    morphK = property(**morphK())

    def minROIArea():
        doc = "TheminROIArea property."
        def fget(self):
            return self._minROIArea
        def fset(self, value):
            self._minROIArea = value
        def fdel(self):
            del self._minROIArea
        return locals()
    minROIArea = property(**minROIArea())

    def maxROIArea():
        doc = "The maxROIArea property."
        def fget(self):
            return self._maxROIArea
        def fset(self, value):
            self._maxROIArea = value
        def fdel(self):
            del self._maxROIArea
        return locals()
    maxROIArea = property(**maxROIArea())

    def epsilon():
        doc = "The epsilon property."
        def fget(self):
            return self._epsilon
        def fset(self, value):
            self._epsilon = value
        def fdel(self):
            del self._epsilon
        return locals()
    epsilon = property(**epsilon())


    def finalSize():
        doc = "The finalSize property."
        def fget(self):
            return self._finalSize
        def fset(self, value):
            self._finalSize = value
        def fdel(self):
            del self._finalSize
        return locals()
    finalSize = property(**finalSize())

    def minTextY():
        doc = "The _minTextY property."
        def fget(self):
            return self._minTextY
        def fset(self, value):
            self._minTextY = value
        def fdel(self):
            del self._minTextY
        return locals()
    minTextY = property(**minTextY())

    def maxTextY():
        doc = "The_maxTextY property."
        def fget(self):
            return self._maxTextY
        def fset(self, value):
            self._maxTextY = value
        def fdel(self):
            del self._maxTextY
        return locals()
    maxTextY = property(**maxTextY())

    def minBlobArea():
        doc = "The _minBlobArea property."
        def fget(self):
            return self._minBlobArea
        def fset(self, value):
            self._minBlobArea = value
        def fdel(self):
            del self._minBlobArea
        return locals()
    minBlobArea = property(**minBlobArea())

    def maxBlobArea():
        doc = "The_maxBlobArea property."
        def fget(self):
            return self._maxBlobArea
        def fset(self, value):
            self._maxBlobArea = value
        def fdel(self):
            del self._maxBlobArea
        return locals()
    maxBlobArea = property(**maxBlobArea())

    def blobDelta():
        doc = "The _blobDelta property."
        def fget(self):
            return self._blobDelta
        def fset(self, value):
            self._blobDelta = value
        def fdel(self):
            del self._blobDelta
        return locals()
    blobDelta = property(**blobDelta())

    def threshBlock():
        doc = "The_threshBlock property."
        def fget(self):
            return self._threshBlock
        def fset(self, value):
            self._threshBlock = value
        def fdel(self):
            del self._threshBlock
        return locals()
    threshBlock = property(**threshBlock())

    def threshC():
        doc = "The_threshC property."
        def fget(self):
            return self._threshC
        def fset(self, value):
            self._threshC = value
        def fdel(self):
            del self._threshC
        return locals()
    threshC = property(**threshC())

    def threshErode():
        doc = "The_threshErode property."
        def fget(self):
            return self._threshErode
        def fset(self, value):
            self._threshErode = value
        def fdel(self):
            del self._threshErode
        return locals()
    threshErode = property(**threshErode())

    def maxE():
        doc = "Thee property."
        def fget(self):
            return self._maxE
        def fset(self, value):
            self._maxE = value
        def fdel(self):
            del self._maxE
        return locals()
    maxE = property(**maxE())

    def maxSymbolCost():
        doc = "The_maxSymbolCost property."
        def fget(self):
            return self._maxSymbolCost
        def fset(self, value):
            self._maxSymbolCost = value
        def fdel(self):
            del self._maxSymbolCost
        return locals()
    maxSymbolCost = property(**maxSymbolCost())
