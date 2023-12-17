import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import os
import time as t



def createMask(clstImg, matrix, startFromHighest=True, NthColor=0):
    if not startFromHighest:
        matrix = matrix[::-1]

    resultImg = clstImg.copy()
    
    lowColor = np.array( matrix[0] )
    highColor = matrix[NthColor]
    highColor = np.array( highColor )

    boards = [lowColor,highColor]
    if startFromHighest:
        boards = boards[::-1]
        
    print('From {} to {}'.format(boards[0],boards[1]))
    mask = cv.inRange(resultImg, boards[0],boards[1])
    return mask

def getNucleiCenterPoints(nuclMask):
    def calcCenter(cnt):
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return [cx,cy]
    
    _, nuclMask = cv.threshold(nuclMask,127,255,cv.THRESH_BINARY)
    contours ,_ = cv.findContours(nuclMask, cv.RETR_LIST,
                               cv.CHAIN_APPROX_SIMPLE)

    pointsArr = []
    for c in contours:
        if len(c)>5:
           pointsArr.append( calcCenter(c) )

    return pointsArr


def filterContours(mask):
    returnMask = np.zeros(mask.shape,dtype=np.uint8)
    _, mask = cv.threshold(mask,127,255,cv.THRESH_BINARY)
    contours ,_ = cv.findContours(mask, cv.RETR_LIST,
                               cv.CHAIN_APPROX_SIMPLE)

    areaArr=[]
    for c in contours:
        if len(c) > 5 and cv.contourArea(c)>5:
            areaArr.append(cv.contourArea(c))
            cv.drawContours(returnMask, [c], -1,
                            255, -1)

    print(np.mean(areaArr))
    print(min(areaArr))
    print(max(areaArr))
    return returnMask

def kMeansAnl(img, clustersNumber):
    K = clustersNumber
    
    (h,w) = img.shape[:2]
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    clt = KMeans(n_clusters = K , n_init = 'auto')
    
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant_K5 = quant.reshape((h, w, 3))
    
    img = img.reshape((h, w, 3))
   
    quant_K5 = cv.cvtColor(quant_K5, cv.COLOR_BGR2GRAY)
    image_K5 = Image.fromarray(quant_K5)
    
    color_matrix = image_K5.getcolors()
    
    color_matrix = [cm[1] for cm in color_matrix]
    print('Color matrix: ',color_matrix)
    color_matrix = np.sort(color_matrix)[::-1] 

    img = np.zeros((image_K5.size[0], image_K5.size[1], 3))
    img = np.array(image_K5).copy()
    
    return img, color_matrix



import pickle

def runClassifier(im,mask,model):
    def getParams(cnt):
        def func_1(x): 
            Area = cv.contourArea(x)

            
            Perimeter = cv.arcLength(x, closed=True)
            if Perimeter == 0:
                return 1

            return round(4*np.pi*Area/(Perimeter**2) , 5)

        def func_2(x): 
            def getMAdummy(cntr):
                xMin = tuple(cnt[cnt[:,:,0].argmin()][0])
                xMax = tuple(cnt[cnt[:,:,0].argmax()][0])
                yMin = tuple(cnt[cnt[:,:,1].argmin()][0])
                yMax = tuple(cnt[cnt[:,:,1].argmax()][0])
                print(xMin,xMax,yMin,yMax)
                distX,distY = xMax-xMin,yMax-yMin
                if distX > distY:
                    return distX
                else:
                    return distY
                
            Area = cv.contourArea(x)
            if len(x) > 10:
                MA = cv.fitEllipse(x)[1][0]
            else:
                return 1
                
            return round(4*Area/(np.pi*MA*MA), 5)

        def func_3(x):
            if len(x)<10:
                return 1
            CH = cv.convexHull(x)
            Area = cv.contourArea(x)
            ConvexArea = cv.contourArea(CH)
            return round(Area/ConvexArea, 5)

        def func_4(x): 
            return round(cv.contourArea(x), 5)

        paramsList = []

        p1 = func_1(cnt)
        p2 = func_2(cnt)
        p3 = func_3(cnt)
        #p4 = func_4(cnt)####

        paramsList.append(p1)
        paramsList.append(p2)
        paramsList.append(p3)
        #paramsList.append(p4)####

        return paramsList
    
        
    contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)


    drawnIm = im.copy()
    
    colors = [[255,0,0],[0,255,0]]
    for c in contours:
        params = getParams(c)
        y = model.predict([params])[0]

        cv.drawContours(drawnIm, [c], -1,
                        colors[y-1], -1)

    return drawnIm

        

    
    
name = '022_10'
img = cv.imread(r'C:\liver\LIVER_CROPS\0_crop.jpg')
#img = cv.bilateralFilter(img, 15, 75, 75)
#cv.imwrite(r'C:\liver\LIVER_crops\afterBLfilter.jpg' , img)
          
clustersNumber = 5


startTime = t.time()
clusteredImg, colorMatrix = kMeansAnl(img, clustersNumber)

whiteRegionsMask = createMask(clusteredImg, colorMatrix,
                              startFromHighest=True, NthColor = 0)

nuclMask = createMask(clusteredImg, colorMatrix,
                      startFromHighest=False, NthColor = 0)

elapsedTime = t.time() - startTime

print('Elapsed time = ', elapsedTime)
cv.imwrite(r'C:\liver\LIVER_CROPS\0_crop_mask.bmp', nuclMask)

'''cv.imwrite(r'C:\\liver\\LIVER_crops\\Preobr181195\\'+str(name)+str('_mask.jpg') , nuclMask)'''

######################

modelPath = 'C:\liver\liver_code\Obrab'
if os.path.exists(modelPath):
    model = pickle.load(open(modelPath, 'rb'))
    drawnIm = runClassifier(img, whiteRegionsMask, model)


    nuclCntrs,_ = cv.findContours(nuclMask,cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    for c in nuclCntrs:
        cv.drawContours(drawnIm, [c], -1, (255,255,255), -1)

    cv.imwrite(r'C:\liver\LIVER_crops\Preobr181195\DRAWN_CNTRS.png',
               drawnIm)























