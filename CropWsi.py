import os
os.add_dll_directory(r'C:\VIPS\bin')

import SlicesDetector as sd
import openslide as wsi

import numpy as np
import cv2 as cv

def imgMask(img):
    blur = cv.GaussianBlur(img,(5,5),0)
    blur = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    return thresh

def show(img , name='img'):
    cv.imshow(name, img)
    cv.waitKey(0)

def getCooridnates(img, num):
    contours, _  = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = contours[num]
    x,y,w,h = cv.boundingRect(cnt)

    return x,y,w,h

def getImgForCrop( mapBinary , wsiObject, number , cropZoom , mapZoom,  marginPercent = 0):
    xStart, yStart, width, height = getCooridnates( mapBinary , number )
    sizeScale = pow(2,mapZoom - cropZoom)
    # !!! in read_region (xStart,yStart) must be in zoom=0  resolution
    locationScale = pow(2, mapZoom)

    xStartCrop = xStart*locationScale
    yStartCrop = yStart*locationScale
    widthCrop = width*sizeScale
    heightCrop = height*sizeScale
    
    if marginPercent > 0:
       widthCrop = int(widthCrop + widthCrop*marginPercent)
       heightCrop = int(heightCrop + heightCrop*marginPercent)

       xStartCrop = int(xStartCrop - widthCrop*marginPercent)
       yStartCrop = int(yStartCrop - heightCrop*marginPercent)

    img = wsiObject.read_region( ( xStartCrop,yStartCrop ),
                                   cropZoom,
                                 ( widthCrop,heightCrop ) )


    img = np.array(img)
    img = cv.cvtColor(img , cv.COLOR_RGBA2BGR)

    return img

def invert(image,name):
    image = (255-image)
    res  = cv2.imwrite(name, image)
    return  res

def findWhite(img):
    GREYimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     
    _,mask = cv.threshold(GREYimg,thresh=180, maxval= 255,  type=cv.THRESH_BINARY)
    THRgrey = cv.bitwise_and(GREYimg,mask)
    
    BlueLevel = 255
    RedLevel =255
    GreenLevel = 255

    THRmask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    threshCOLOR = cv.bitwise_and(img,THRmask)

    return threshCOLOR


wsiPath = r"C:\liver\015.mrxs"
liverSampleName = "015"
reportBasePath = r"C:\\liver\\LIVER_CROPS\\"

wsiObj = wsi.OpenSlide(wsiPath)

mapZoom = 8
cropZoom = 3
cropNum = 10

slicesDetector = sd.SlicesDetector(wsiObj, mapZoom)

MAP = slicesDetector.mapImg

mapBinary = imgMask(MAP)
cv.imwrite(reportBasePath+'mapBinar.jpg' , mapBinary)

marginPercent = 0.1 # процент отступа

countoursCount, _ = cv.findContours(mapBinary,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
NUM  = len(countoursCount)-1

for i in range(NUM):
    singleSLICE = getImgForCrop(mapBinary, wsiObj, i , cropZoom, mapZoom , marginPercent)
    cv.imwrite(reportBasePath+str(i)+ '_' + 'crop.jpg', singleSLICE)    


cropPath = r'C:\\liver\\LIVER_crops\\0_crop.jpg'
selectedCrop = cv.imread( cropPath )

onlyWhiteRegionsImg = findWhite(selectedCrop)
cv.imwrite(r'C:\\liver\\LIVER_crops\\COUNTOURS.jpg' , onlyWhiteRegionsImg)
    



