import cv2 as cv
import numpy as np
from datetime import datetime
import os
from pathlib import Path


class RegionSelector:
    def __init__(self, imgPath, maskPath, needResize=False):
        self.imPath = imgPath
        self.mPath = maskPath

        self.divideMode = False
        
        self.origImg = cv.imread(imgPath)
        self.mask = cv.imread(maskPath,0)
        
        self.resizeScale = 1
        self.imgW, self.imgH = self.origImg.shape[1],self.origImg.shape[0]
        if needResize:
            from win32api import GetSystemMetrics

            scale = 0
            userW, userH = GetSystemMetrics(0),GetSystemMetrics(1)
            userH = userH - 100 # из-за размеров панели задач и шапки окна с картинкой
            imgW,imgH = self.imgW, self.imgH
            if imgW<userW and imgH<userH:
                diffW = userW-imgW
                diffH = userH-imgH
                if diffW>diffH:
                    scale = userH/imgH
                if diffH>diffW:
                    scale = userW/imgW
                    
            if imgW>userW and imgH>userH:
                diffW = imgW-userW
                diffH = imgH-userH
                if diffW>diffH:
                    scale = userH/imgH
                if diffH>diffW:
                    scale = userW/imgW

            if scale != 0:
                self.origImg = cv.resize(self.origImg,(0,0),fx=scale,fy=scale)
                self.mask = cv.resize(self.mask,(0,0),fx=scale,fy=scale)
                self.resizeScale = scale
                
                
        self.drawedImg = self.origImg.copy()
        _, self.mask = cv.threshold(self.mask,127,255,cv.THRESH_BINARY)

        self.regionContours = self.getDrawContours()
        self.selectedContoursIndeces = []
        self.contourColor = (0,0,0)
        self.processContourColor = (0,0,255)
        self.unselectedContourSize = 1
        
        self.CNT_NOT_FOUND_FLAG = -1
        self.lastSelectedClass = -1
        self.rounding = 5
        self.PPM = 0.25
        
        self.reportFileName = 'RegSelectReport_'+Path(imgPath).stem+'_'+datetime.now().strftime("%d.%m.%Y %H-%M-%S")+'.txt'
        self.reportFileName = os.getcwd()+'\\'+self.reportFileName
        self.reportDelimeter = ','
        self.createReportFile()


    def setCallback(self):
        cv.setMouseCallback("Region selector", self.processClick)

    def unsetCallback(self):
        cv.setMouseCallback("Region selector", lambda *args : None)
        
    def run(self):
        # window must be created after all defenitions
        cv.namedWindow("Region selector", cv.WINDOW_AUTOSIZE)
        self.updateCanvas(needCalc=False)
        self.setCallback();
        

    def addNewReportLine(self, contourCalcParams):
        with open(self.reportFileName, 'a', encoding='utf-8') as repFile:
            for ind, calcParam in enumerate(contourCalcParams):
                repFile.write(str(calcParam))
                if ind != len(contourCalcParams)-1:
                    repFile.write(self.reportDelimeter)
            repFile.write('\n')


    def createReportFile(self):
        f = open(self.reportFileName, 'a')
        headerParams = {'img path': self.imPath,
                        'mask path': self.mPath,
                        'img width': self.imgW,
                        'img height': self.imgH,
                        'resize scale': self.resizeScale,
                        'ppm': self.PPM}
        
        for headerParamName, value in headerParams.items():
            f.write(headerParamName + '=')
            f.write( str(value) )
            f.write(';')
            

        f.write('\n')
        f.close()

    def findSelectedContour(self, xPos, yPos):
        index = self.CNT_NOT_FOUND_FLAG
        testingPoint = (xPos,yPos)
        for i in range(len(self.regionContours)):
            singleContour = self.regionContours[i]
            pptResult = cv.pointPolygonTest(singleContour,
                                           testingPoint, measureDist=False)
            if pptResult == 1:
                index = i
                break
        return index

    def processClick(self, event, x,y, flag,param):
        if event == cv.EVENT_LBUTTONDOWN:
            cntIndex = self.findSelectedContour(x,y)
            if (cntIndex != self.CNT_NOT_FOUND_FLAG) and (cntIndex not in self.selectedContoursIndeces):
                self.selectedContoursIndeces.append(cntIndex)
                self.unsetCallback()
                self.updateCanvas()

        if event == cv.EVENT_RBUTTONDOWN:
            if len(self.selectedContoursIndeces)!=0:
                self.selectedContoursIndeces = self.selectedContoursIndeces[:-1]
                self.updateCanvas()
            
            

    def getDrawContours(self):
        regionContours , _ = cv.findContours(self.mask,
                                            cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)
        return regionContours

    def drawCnt(self, cnt, colorType, needFill=False):
        fillingType = self.unselectedContourSize
        if needFill: fillingType = -1;
        cv.drawContours(self.drawedImg, cnt, 0, colorType, fillingType)

    def calculateParams(self, cnt):
        paramsList = []

        # return value of functions must be single number
        def func_1(x): # компактность
            Area = cv.contourArea(x)
            Perimeter = cv.arcLength(x, closed=True)
            return round(4*np.pi*Area/Perimeter , self.rounding)

        def func_2(x): # округлость
            def getMAdummy(cntr):
                xMin = tuple(cnt[cnt[:,:,0].argmin()][0])
                xMax = tuple(cnt[cnt[:,:,0].argmax()][0])
                yMin = tuple(cnt[cnt[:,:,1].argmin()][0])
                yMax = tuple(cnt[cnt[:,:,1].argmax()][0])

                distX,distY = xMax-xMin,yMax-yMin
                if distX > distY:
                    return distX
                else:
                    return distY
                
            Area = cv.contourArea(x)
            if len(x) > 4:
                MA = cv.fitEllipse(x)[1][0]
            else:
                MA = getMAdummy(x)
                
            return round(4*Area/(np.pi*MA*MA), self.rounding)

        def func_3(x): # выпуклость
            CH = cv.convexHull(x)
            Area = cv.contourArea(x)
            ConvexArea = cv.contourArea(CH)
            return round(Area/ConvexArea, self.rounding)

        def func_4(x): # площадь
            return round(cv.contourArea(x), self.rounding)
        

        paramsList.append( func_1(cnt) )
        paramsList.append( func_2(cnt) )
        paramsList.append( func_3(cnt) )
        paramsList.append( func_4(cnt) )

        # put center coordinate for future working
        M = cv.moments(cnt)
        centerX = int(M['m10']/M['m00'])
        centerY = int(M['m01']/M['m00'])
        paramsList.append( centerX )
        paramsList.append( centerY )

        return paramsList

    def waitForContourClass(self):
        key = cv.waitKey(0)
        for CLASS in self.CLASSES:
            if key == ord(CLASS):
                self.lastSelectedClass = CLASS
                return False

            if key == 27:
                self.stopProgram()

        return True
 
    def updateCanvas(self, needCalc=True):
        self.drawedImg = []
        self.drawedImg = self.origImg.copy()
        for i in range(len(self.regionContours)):
            cnt = self.regionContours[i]
            if i not in self.selectedContoursIndeces:       #unselected contour
                self.drawCnt([cnt], self.contourColor)
            else:
                if i == self.selectedContoursIndeces[-1]:   #just now selected contour
                    processingContourIndex = i
                    self.drawCnt([cnt], self.processContourColor, True)
                else:                                       #selected contour earlier
                    self.drawCnt([cnt], self.contourColor, True)
    
        cv.imshow("Region selector", self.drawedImg)
        
        if needCalc:

            while self.waitForContourClass():
                pass

            cnt = self.regionContours[processingContourIndex]
        
            paramsList = self.calculateParams( cnt )
            paramsList.append( self.lastSelectedClass )
            self.addNewReportLine( paramsList )

            self.drawCnt([cnt], self.contourColor, True)
            cv.imshow("Region selector", self.drawedImg)
            self.setCallback()

    def stopProgram(self):
        
        def isSmallCnt(x):
            if len(x) < 0:
                return True
            return False
        
        DIVIDE_CLASS = 2
        allContours = self.regionContours
        
        for ind,cnt in enumerate(allContours):
            if ind not in self.selectedContoursIndeces:
                if not isSmallCnt(cnt):
                    paramsList = self.calculateParams( cnt )
                    paramsList.append( DIVIDE_CLASS )
                    self.addNewReportLine( paramsList )

        cv.destroyAllWindows()
        cv.waitKey(0)
        
        


#################################################################################

imgPath = r'LIVER.png'
maskPath = r'saturMask.jpg'
needResize = False


regSelector = RegionSelector(imgPath, maskPath, needResize)
regSelector.CLASSES = ['1', '2']
regSelector.PPM = 0.25 # не ебу, тут на вскидку. надо во вьюере смотреть под каждый ВС желательно
                       # конечно, параметр не принципиальный, но, например,
                       # при использовании площади как параметра это важно,
                       # т.к. ppm может отличаться

''' 1: маленькие области '''
''' 2: большиие области, и т.д. '''
''' можно указать любой символ на клавиатуре - он будет меткой класса,
            которая пишется в файл отчёта отдельной колонкой '''


regSelector.divideMode = True

regSelector.run()








