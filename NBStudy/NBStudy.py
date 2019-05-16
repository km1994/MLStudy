# coding:UTF-8
import numpy as np
from collections import defaultdict
class NBClass():

    def __init__(self):
        self.dataMat=[]
        self.labelMat=[]
        self.testArr=[]
        self.trainDic=defaultdict(int)
        self.labelDic = defaultdict(int)
        self.labelPDict=defaultdict(int)
        self.trainDataPPDict=defaultdict(int)
        self.NBDic = defaultdict(int)
        self.argMaxVal = 0
        self.argMaxKey = ''

    #######################################
    #加载数据
    #input:  path 训练数据
    #output: dataMat(list)特征
    #       labelMat(list)标签
    #######################################
    def loadTxtData(self,path):
        print("-----------loadTxtData-----------")
        dataMat = []
        labelMat = []
        fr = open(path)  # 打开文件
        for line in fr.readlines():
            lines = line.strip().split("\t")
            lineArr = []

            for i in range(len(lines) - 1):
                lineArr.append(lines[i])
            dataMat.append(lineArr)

            labelMat.append(lines[-1])  # 转换成{-1,1}
        fr.close()
        return dataMat, labelMat

    #######################################
    #统计训练数据中，新样例所占数量
    # eg :('Sunny', 'No'): 3
    #input:  testArr(list) 目标新列表
    #output: testArr(list) 目标新列表
    #output: trainDict(dic) 训练数据中，新样例所占数量
    #######################################
    def trainDataCount(self,testArr):
        print("-----------trainDataCount-----------")
        trainDict=defaultdict(int)
        for i in range(len(self.getDataTrain())):
            for (a,b) in zip(self.getDataTrain()[i],self.getLabelTrain()):
                if a == testArr[i]:
                    trainDict[(a,b)]=trainDict[(a,b)]+1
        self.setTestArr(testArr)
        print(trainDict)
        self.setTrainDic(trainDict)

    #######################################
    # 统计标签数量
    # eg :({'No': 0.35714285714285715, 'Yes': 0.6428571428571429}
    # output: labelDic(dic) 标签数量字典
    #######################################
    def labelDataCount(self):
        print("-----------labelDataCount-----------")
        labelSets=set(self.getLabelTrain())
        labelDic={}
        i=0
        for labelSet in labelSets:
            labelDic[labelSet]=len([item for item in self.getLabelTrain() if item == labelSet])
            i=i+1
        self.setLabelDic(labelDic)

    #######################################
    # 事件A的先验概率（prior probability）
    # eg :({'No': 0.35714285714285715, 'Yes': 0.6428571428571429}
    # output: labelDic(dic) 标签数量字典
    #######################################
    def calPriorProbability(self):
        print("-----------calPriorProbability-----------")
        labelPDict={}
        for (key,value) in self.getLabelDic().items():
            labelPDict[key]=self.getLabelDic().get(key)/self.getAllLabel()
        self.setLabelPDict(labelPDict)
        print(self.getLabelPDict())

    #######################################
    # 事件A的后验概率（prior probability）
    # eg :{('Sunny', 'No'): 0.6, ('Sunny', 'Yes'): 0.2222222222222222, ('Cool', 'Yes'): 0.3333333333333333, ('Cool', 'No'): 0.2, ('High', 'No'): 0.8, ('High', 'Yes'): 0.3333333333333333, ('Strong', 'No'): 0.6, ('Strong', 'Yes'): 0.3333333333333333}
    # eg: p('Sunny'|'No')=0.6
    # output: trainDataPPDict(dic)  后验概率字典
    #######################################
    def calPosteriorProbability(self):
        print("-----------calPosteriorProbability-----------")
        trainDataPPDict={}
        for ((key11,key12),value1) in self.getTrainDic().items():
            #print(self.getLabelDic()[key12])
            #print(((key11, key12), value1))
            trainDataPPDict[(key11, key12)] = value1 / self.getLabelDic()[key12]
        print(trainDataPPDict)
        self.setTrainDataPPDict(trainDataPPDict)

    #######################################
    # 计算argmax, 即朴素贝叶斯概率最大的那个元素的下标
    # output: argMaxVal(int)  最大概率的值
    # output: argMaxKey(int)  最大概率的下标
    #######################################
    def getArgMax(self):
        print("---------getArgMax--------")
        NBDict=defaultdict(int)
        labelSets = set(self.getLabelTrain())
        for labelSet in labelSets:
            mal=self.getLabelPDict()[labelSet]
            for i in range(len(self.getTestArr())):
                mal=mal*self.getTrainDataPPDict()[(self.getTestArr()[i],labelSet)]
            NBDict[labelSet]=mal
            if self.getArgMaxVal()<mal:
                self.setArgMaxVal(mal)
                self.setArgMaxKey(labelSet)
        print(self.getArgMaxKey(),":",self.getArgMaxVal())

    def main(self,path,testArr):
        print("-----------main-----------")
        self.dataTrain, self.labelTrain = self.loadTxtData(path)
        self.setDataTrain(np.array(self.dataTrain).T)
        self.labelDataCount()
        self.trainDataCount(testArr)
        self.calPriorProbability()
        self.calPosteriorProbability()

        self.getArgMax()

    def getAllLabel(self):
        return len(self.getLabelTrain())
    def setDataTrain(self,dataTrain):
        self.dataTrain=dataTrain
    def getDataTrain(self):
        return self.dataTrain
    def setLabelTrain(self,labelTrain):
        self.labelTrain=labelTrain
    def getLabelTrain(self):
        return self.labelTrain
    def setTestArr(self,testArr):
        self.testArr=testArr
    def getTestArr(self):
        return self.testArr
    def setTrainDic(self,trainDic):
        self.trainDic=trainDic
    def getTrainDic(self):
        return self.trainDic
    def setLabelDic(self,labelDic):
        self.labelDic=labelDic
    def getLabelDic(self):
        return self.labelDic
    def setLabelPDict(self,labelPDict):
        self.labelPDict=labelPDict
    def getLabelPDict(self):
        return self.labelPDict
    def setTrainDataPPDict(self,trainDataPPDict):
        self.trainDataPPDict=trainDataPPDict
    def getTrainDataPPDict(self):
        return self.trainDataPPDict
    def setNBDict(self,NBDict):
        self.NBDict=NBDict
    def getNBDict(self):
        return self.NBDict
    def setArgMaxVal(self,argMalVal):
        self.argMaxVal=argMalVal
    def getArgMaxVal(self):
        return self.argMaxVal
    def setArgMaxKey(self,argMalKey):
        self.argMalKey=argMalKey
    def getArgMaxKey(self):
        return self.argMalKey

if __name__ == "__main__":
    nbObj=NBClass()
    testArr=['Sunny','Cool','High','Strong']
    nbObj.main("playTennis.txt",testArr)