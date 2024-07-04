import pandas as pd
import time


def makeModel(w, b):
    def model(x):
        return (w * x + b)
    return model

def modelCost(w, b, df, x, y): 
    model = makeModel(w, b)
    totalCt = 0
    totalCost = 0
    for (index, row) in df.iterrows():
        totalCt += 1
        littleCost = ((model(row[x]) - row[y])**2)
        totalCost += littleCost
        print(f'total cost: {totalCost}; index: {index}; ct: {totalCt}; current cost: {littleCost}; bmi: {row[x]}; predicted: {model(row[x])}; realcost: {row[y]}; difference: {model(row[x]) - row[y]}')
        time.sleep(0.001)
        
    return (totalCost / totalCt)

def dW(w, b, df, x, y):
    model = makeModel(w, b)
    totalCt = 0
    totalDw = 0

    for (index, row) in df.iterrows():
        totalCt += 1
        littleDw = (2 * w * (model(row[x]) - row[y]))
        totalDw +=  littleDw
    return (totalDw / totalCt)

def dB(w, b, df, x, y):

    totalCt = 0
    totalDb = 0

    for (index, row) in df.iterrows():
        totalCt += 1
        totalDb += (2 * (w * row[x] + b - row[y]))

    return totalDb
    

def gradientDescent(alpha, df, x, y):

    dfNorm = df.copy()
    xMax = dfNorm[x].abs().max()
    yMax = dfNorm[y].abs().max()

    dfNorm[x] = dfNorm[x] / xMax
    dfNorm[y] = dfNorm[y] / yMax
    
    dw = 1
    db = 1
    w = 1
    b = 1
    currentCost = modelCost(w, b, dfNorm, x, y)
    while (dw > 0.001 and db >  0.001):
        dw = dW(w, b, dfNorm, x, y)
        db = dB(w, b, dfNorm, x, y)
        
        w -= alpha * dw
        b -= alpha * db
        
        newCurrentCost = modelCost(w, b, dfNorm, x, y)
        
        print(newCurrentCost)
        print(currentCost)
        print(f'w: {w}; b: {b}')

        if (newCurrentCost > currentCost):
            print("it broke")
            return
        currentCost = newCurrentCost
        


    print("it workt")
    w = w * yMax / xMax
    b = b * yMax

    print(f'w: {w}; b: {b}')
    return (w, b)

insurance = pd.read_csv('./insurance.csv')





print(gradientDescent(0.0005, insurance, 'bmi', 'charges'))

