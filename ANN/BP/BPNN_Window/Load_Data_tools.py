from numpy import *


def loadDataSet( filename):  # 加载数据集
    dataMat = []
    classLabels = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMatItem=[]
        trainData=[]
        trainData.append(float(lineArr[0]))
        trainData.append(float(lineArr[1]))
        dataMatItem.append(trainData)

        classLabels=[]
        classLabels.append(int(lineArr[2]))

        dataMatItem.append(classLabels)
        dataMat.append(dataMatItem)
    return dataMat

def switch_load_data_fun(type,address="./../testSet2.txt"):
    if type == 1:
        dataMat = loadDataSet("./../testSet2.txt")
    elif type == 2:
        print("address: ",address)
        dataMat = loadDataSet(address)
    else:
        pass
    return dataMat


#print("dataMat:",dataMat)