#This files contains thresholds to tune the HLD_Helper
import cv2
import numpy as np

class Tuner:
    """
        Hazard labels detector tuner
    """

    def __init__(self):
        #pre-processing
        self._medianKSize = 5
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
        self._textCropX   = 25
        self._textCropY   = 120
        self._minBlobArea = 125
        self._maxBlobArea = 4000
        self._threshBlock = 33
        self._threshC     = 2
        self._maxE        = 0.98
        self._minTextHeight = 20

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

    def textCropY():
        doc = "The textCropY property."
        def fget(self):
            return self._textCropY
        def fset(self, value):
            self._textCropY = value
        def fdel(self):
            del self._textCropY
        return locals()
    textCropY = property(**textCropY())

    def textCropX():
        doc = "The textCropX property."
        def fget(self):
            return self._textCropX
        def fset(self, value):
            self._textCropX = value
        def fdel(self):
            del self._textCropX
        return locals()
    textCropX = property(**textCropX())

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

    def minTextHeight():
        doc = "TheminTextHeight property."
        def fget(self):
            return self._minTextHeight
        def fset(self, value):
            self._minTextHeight = value
        def fdel(self):
            del self._minTextHeight
        return locals()
    minTextHeight = property(**minTextHeight())