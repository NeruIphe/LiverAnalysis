import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def show1chHist(img):
    plt.hist(img.ravel(),
             256,
             [0,256])
    plt.show()


name = 'im1'

imgBASED = cv.imread(r'C:\\liver\\LIVER_crops\\'+name+'.jpg')
imgOrig = cv.imread(r'C:\\liver\\LIVER_crops\\'+name+str(r'__181__195')+'.jpg')
WHITE_THRESHOLD = 180 # подобрано. Надо будет калиброваться по пустому месту на стекле

img = cv.GaussianBlur(imgOrig, (7, 7), 0)
labImg = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# разбиваем картинку lab на 3 канала
lChannel,aChannel,bChannel = cv.split(labImg)
lChannel = cv.equalizeHist(lChannel)

kernelSize = 11
#show1chHist(lChannel)

# проводим пороговую сегментацию по каналу яркости
thrVal, lMask = cv.threshold(lChannel,
                        WHITE_THRESHOLD,
                        255,
                        cv.THRESH_BINARY)

cv.imwrite('C:\liver\LIVER_crops\lMask.jpg' , lMask)
cv.imwrite('C:\liver\LIVER_crops\liverRESULT.jpg' , cv.bitwise_and(imgOrig,imgOrig,mask=lMask))

contours, _ = cv.findContours(lMask,
                                    cv.RETR_TREE,
                                    cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(imgBASED, contours, -1, (0,0,0), 1 )

cv.imwrite('C:\liver\LIVER_crops\liverDRAWEDcntrs.jpg' , imgBASED)















    
