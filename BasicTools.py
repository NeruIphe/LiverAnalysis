from numpy import sign, linspace, argmin, ogrid, argmax, array, where

def medianFilter(arr, window=3):
    if window < 1 or window > len(arr):
        raise Exception("Invalid window param")
    for i in range(0,len(arr)-window+1):
        arr[i] = sum(arr[i:i+window])/window
    return arr[:len(arr)-window+1]

def getNearestIndex(arr,val):
    return argmin( abs(array(arr)-val) )

def findExtremum(arr, numApprovePoints=5):
    accumulPoints = 0
    trendPoints = []
    trendDirection = 0
    trendApproved = False
    for i in range(1,len(arr)):
        if sign(arr[i]-arr[i-1]) == trendDirection:
            if not trendApproved: accumulPoints += 1;
        else:
            accumulPoints = 0
            trendApproved = False
            
        trendDirection = sign(arr[i]-arr[i-1])
        if accumulPoints==numApprovePoints:
            trendPoints.append( (trendDirection, i-numApprovePoints) )
            trendApproved = True

    result = []
    trendSize = 0
    firstIndex, lastIndex = 0,0

    trendPoints.append( (None,None) ) # for working with last trend       
    for i in range(1,len(trendPoints)):
        direct, index = trendPoints[i]
        if direct == trendPoints[i-1][0]:
            trendSize+=1
            lastIndex = trendPoints[i][1]
        else:
            '''
            result.append( {'trendDirection':   trendPoints[i-1][0],
                            'firstIndex':       firstIndex,
                            'lastIndex':        lastIndex,
                            'trendSize':        trendSize} )
            '''
            result.append( (trendPoints[i-1][0],
                            firstIndex,
                            lastIndex,
                            trendSize) )
            trendSize = 0
            firstIndex = trendPoints[i][1]
             
    return array(result)

def selectFromDictsArr(arr, key, func_):
    res=[]
    res = list([singleDict[key] for singleDict in arr])
    return func_(res)

def moreZero(arr):
    return where(array(arr)>0)
        