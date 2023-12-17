from numpy import array, mean, diff, median, var, where, absolute, polyfit, ones, uint8
import cv2 as cv
from matplotlib import pyplot as plt
from BasicTools import *


############################################
    
class SlicesDetector:
    def __init__(self, wsiObj, mapZoom, mode='gaussAdapt'):
        self.thrType = cv.THRESH_BINARY_INV
        self.medianFilterWindow = 7
        self.numApproveTrendPoints = 5
        self.useOpenSlideMap = True
        self.showPlots = False
        
        
        self.wsiObject = wsiObj
        self.minZoom = mapZoom
        

        self.mapWidth,self.mapHeight = self.wsiObject.level_dimensions[self.minZoom]

        self.mapImg = self.getMapImage()
        if self.useOpenSlideMap:
            self.mapImg = self.removeBlack(self.mapImg) #need only 3 channels image
        
        self.mapGray = cv.cvtColor(self.mapImg, cv.COLOR_BGR2GRAY)

        if mode == 'customAdapt':
            _, self.mapBinary = cv.threshold(self.mapGray,
                                          self.calcOptimalThrVal(),
                                          255, self.thrType)

            self.mapBinary = self.postProcMap(self.mapBinary)

        if mode == 'gaussAdapt':
            kernel = ones((5,5),uint8)
            thrGauss = cv.adaptiveThreshold(self.mapGray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV,5,2)

            #thrGauss = cv.erode(thrGauss, (2,2))

            cv.imwrite('pureGauss.jpg' , thrGauss)
            
            thrGauss = cv.morphologyEx(thrGauss, cv.MORPH_CLOSE, kernel)
            cv.imwrite('closeGauss.jpg' , thrGauss)
            thrGauss = cv.morphologyEx(thrGauss, cv.MORPH_OPEN, kernel)
            cv.imwrite('closeOpenGauss.jpg' , thrGauss)

            self.mapBinary = thrGauss
            
            

        self.slicesAmount = 0


    def postProcMap(self, mapBW):
        return mapBW
        kSize = 5*(self.wsiObject.level_count - self.minZoom)
        kernel = ones((kSize,kSize),uint8)
        mapBW = cv.dilate(mapBW,kernel,iterations = 1)
        return mapBW
        for operation in [cv.MORPH_OPEN,cv.MORPH_CLOSE]:
            mapBW = cv.morphologyEx(mapBW, operation, kernel)
        return mapBW

    def removeBlack(self, img):
        blackPixels = where( (img[:,:,0]< 15) &
                             (img[:,:,1]< 15) &
                             (img[:,:,2]< 15) )
        img[blackPixels] = [255,255,255]
        return img
    
    def getMapImage(self):
        if self.useOpenSlideMap:
            mapIm = self.wsiObject.get_thumbnail((self.mapWidth,self.mapHeight))
            return array(mapIm)
        else:
            mapIm = self.wsiObject.read_region((0,0),
                                                self.minZoom,
                                               (self.mapWidth,self.mapHeight))
            mapIm = array(mapIm)
            mapIm = cv.cvtColor(mapIm, cv.COLOR_BGRA2BGR)
            
            return mapIm

    def findOptimalMinExtremum(self, trendRes):
        positiveTrendRowIndeces = where(trendRes[:,0]>0)
        
        maxPosTrendSizeSubindex = argmax(trendRes[positiveTrendRowIndeces , 3])
        maxPosTrendSizeIndex = positiveTrendRowIndeces[0][maxPosTrendSizeSubindex]
        
        trendRes = trendRes[maxPosTrendSizeIndex:,:]
        negativeTrendRowIndeces = where(trendRes[:,0]<0)
        
        maxNegTrendSizeSubindex = argmax(trendRes[negativeTrendRowIndeces , 3])
        maxNegTrendSizeIndex = negativeTrendRowIndeces[0][maxNegTrendSizeSubindex]
        
        optimalMinExtrValue = trendRes[maxNegTrendSizeIndex][2] # 2= last 0-256 index
        return optimalMinExtrValue

    def calcOptimalThrVal(self):
        brightnessArr = []
        thrValStep = 1
        for thrVal in range(0, 251-thrValStep, thrValStep):
            _, thrImg = cv.threshold(self.mapGray, thrVal, 255, self.thrType)
            mapBrightness = mean(thrImg)
            brightnessArr.append( mapBrightness )

        diffBrArr = diff(brightnessArr)
        diffBrArr = medianFilter(diffBrArr, self.medianFilterWindow)

        trendResult = findExtremum(diffBrArr, self.numApproveTrendPoints)
        optimalThrVal = self.findOptimalMinExtremum(trendResult)
        optimalThrVal += self.medianFilterWindow - 1
        
        if self.showPlots:
            plt.plot( range(len(diffBrArr)), diffBrArr )
            plt.axvline( optimalThrVal, color='r')
            plt.show()
        return optimalThrVal+10

    def getSlices(self):
        for i in range(self.slicesAmount):

            yield i


##########################################################

class SlicesDetector_local:

    def __init__(self, path):
        self.thrType = cv.THRESH_BINARY_INV
        self.medianFilterWindow = 7
        self.numApproveTrendPoints = 5
        self.showPlots = True

        self.mapImg = cv.imread( path )
        self.mapGray = cv.cvtColor(self.mapImg, cv.COLOR_BGR2GRAY)
        _, self.mapBinary = cv.threshold( self.mapGray,
                                          self.calcOptimalThrVal(),
                                          255, self.thrType)
        self.mapBinary = postProcMap(self.mapBinary)

    def postProcMap(self, origMap):
        kernel = np.ones((11, 11), np.uint8)
        return cv2.morphologyEx(origMap, cv2.MORPH_CLOSE, kernel, iterations=3)

    def calcOptimalThrVal(self):
        brightnessArr = []
        thrValStep = 1
        for thrVal in range(0, 251-thrValStep, thrValStep):
            _, thrImg = cv.threshold(self.mapGray, thrVal, 255, self.thrType)
            mapBrightness = mean(thrImg)
            brightnessArr.append( mapBrightness )

        diffBrArr = diff(brightnessArr)
        diffBrArr = medianFilter(diffBrArr, self.medianFilterWindow)

        trendResult = findExtremum(diffBrArr, self.numApproveTrendPoints)
        optimalThrVal = self.findOptimalMinExtremum(trendResult)
        if self.showPlots:
            plt.plot( range(len(diffBrArr)), diffBrArr )
            plt.axvline( optimalThrVal, color='r')
            plt.show()
        return optimalThrVal

    def findOptimalMinExtremum(self, trendRes):
        positiveTrendRowIndeces = where(trendRes[:,0]>0)
        
        maxPosTrendSizeSubindex = argmax(trendRes[positiveTrendRowIndeces , 3])
        maxPosTrendSizeIndex = positiveTrendRowIndeces[0][maxPosTrendSizeSubindex]
        
        trendRes = trendRes[maxPosTrendSizeIndex:,:]
        negativeTrendRowIndeces = where(trendRes[:,0]<0)
        
        maxNegTrendSizeSubindex = argmax(trendRes[negativeTrendRowIndeces , 3])
        maxNegTrendSizeIndex = negativeTrendRowIndeces[0][maxNegTrendSizeSubindex]
        
        optimalMinExtrValue = trendRes[maxNegTrendSizeIndex][2] # 2= last 0-256 index
        return optimalMinExtrValue
    




