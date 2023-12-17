import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pickle


def getReportFilesList(folderPath):
    fileList = []
    for name in os.listdir(folderPath):
        if name[-3:] == 'txt':
            fileList.append(name)
    return fileList


'''
reportFolderPath = r'repFolder'
filesList = getReportFilesList(reportFolderPath)


columnNamesOrig = ['Компактность','Округлость','Выпуклость','Площадь','Xc','Yc','class']

mergedData = []
for fName in filesList:
    path2df = os.path.join(reportFolderPath,fName)
    mergedData.append( pd.read_table(path2df,
                                     skiprows=1,
                                     names=columnNamesOrig,
                                     sep=',') )


mergedData = pd.concat(mergedData)
mergedData.to_csv(reportFolderPath+'\\mergedDF.csv', index=False)
'''

#columnNamesOrig = ['Компактность','Округлость','Выпуклость','Площадь','Xc','Yc','class']

columnNamesOrig = ['Компактность',
                   'round-factor',
                   'Округлость',
                   'blueM',
                   'blueStd',
                   'redM',
                   'redStd',
                   'Xc','Yc',
                   'class']


needSkipFirstRow = 0

reportTablePath = r'C:\LIVER\TrainData_nucls\final_2.txt'
data = pd.read_table(reportTablePath,
                     skiprows = needSkipFirstRow,
                     names = columnNamesOrig,
                     sep = ',')

print( len(data) )

for colName in ['Xc','Yc']:
    del data[colName]

columnNames = ['Компактность',
               'round-factor',
               'Округлость',
               'class']


X = data[ columnNames[:-1] ].to_numpy()
Y = data[ 'class' ].to_numpy()

print(X.shape, Y.shape)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X , Y)

resScore = model.score(X, Y)
print('Score = ',resScore)

x1 = X[392]
x2 = X[393]
x3 = X[394]

def LRfunc(x,p1,p2,p3):
    return x[0]*p1 + x[1]*p2 + x[2]*p3

print('classes',model.classes_)
print('coef',model.coef_)
print('intercept',model.intercept_)
print('n_features',model.n_features_in_)
print('n_iter',model.n_iter_)

'''
weights = model.coef_[0]
print( LRfunc(x1, weights[0],weights[1],weights[2]) )
print( LRfunc(x2, weights[0],weights[1],weights[2]) )
print( LRfunc(x3, weights[0],weights[1],weights[2]) )
'''

print(x1)
print([x1])


print(model.predict( [x1] ))
print(model.predict( [x2] ))
print(model.predict( [x3] ))

pickle.dump(model, open('cellsClassifier_v2', 'wb'))









