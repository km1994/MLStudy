# coding:UTF-8
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class KNNClass():

    def __init__(self):
        pass

    ####################################
    # 1 训练数据加载
    #  data_filename： csv 文件地址
    #  sep：指定分隔符
    #  return dataset  DataFrame格式
    ####################################
    def loadCsv(self,data_filename,sep='\t'):
        dataset = pd.read_csv(data_filename, sep=sep)
        return dataset

    ####################################
    # 1 测试数据加载
    #  data_filename： csv 文件地址
    #  sep：指定分隔符
    #  return dataset  DataFrame格式
    ####################################
    def loadTestCsv(self,data_filename,sep='\t'):
        dataset = pd.read_csv(data_filename, sep=sep)
        return dataset

    ####################################
    # 3 测试数据加载
    #  data_filename： csv 文件地址
    #  sep：指定分隔符
    #  return dataset  DataFrame格式
    ####################################
    def initData(self,dataset):
        dataset['no'] = range(len(dataset))
        dataset['distance'] = 0.0
        return dataset

    ####################################
    # 3.1 欧式距离计算
    #
    ####################################
    def euclideanDistance(self,trainSets,testSets):
        dataCols=trainSets.columns
        for i in range(0,len(trainSets)):
            for col in range(0, len(dataCols) - 3):
                trainSets['distance'][i]=trainSets['distance'][i]+(trainSets[dataCols[col]][i]-testSets[dataCols[col]])**2
                trainSets.loc[i,'distance']=np.sqrt(trainSets.loc[i,'distance'])
        return trainSets

    ####################################
    # 4 根据距离排序，列出前k位
    #
    ####################################
    def sorfDistance(self,dataset,k=5):
        dataset=dataset.sort_values(by=["distance"],ascending=True)
        #print(dataset)
        topK=dataset[:k]
        print(topK)
        return topK

    ####################################
    # 5 根据KNN结果，判断测试集属于哪一个类
    # 功能：通过统计距离测试样本最近的K个训练样本集合的数量，
    #         判断测试样本属于哪一个样本
    #  参数：topK  距离测试样本最近的K个训练样本集合
    #
    ####################################
    def testClass(self,topK):
        categoryTrueNum=topK[(topK['有无']==1)]['有无'].count()
        categoryFalseNum =topK[(topK['有无'] == 0)]['有无'].count()
        if categoryTrueNum> categoryFalseNum:
            return 1
        else:
            return 0

    ####################################
    # 6 主函数
    #
    ####################################
    def main(self,data_filename,dataset,testSets,k):
        dataset=self.initData(dataset)
        dataset=self.euclideanDistance(dataset, testSets)
        topK=self.sorfDistance(dataset,k)
        result=self.testClass(topK)
        return result

    ####################################
    # 7 画原始数据集的散点图
    #
    ####################################
    def drawOrginPic(self,datasets):
        cols=datasets.columns
        self.drawPicCommon(datasets, cols)
        plt.show()

    def drawTestPic(self,datasets,testDataset):
        cols = datasets.columns

        plt.scatter(testDataset[cols[0]], testDataset[cols[1]], c='green', alpha=1, marker='+', label='测试点')  # c='green'定义为绿色
        self.drawPicCommon(datasets,cols)
        plt.show()

    def drawPicCommon(self,datasets,cols):
        print(datasets)
        plt.scatter(datasets[(datasets[cols[2]] == 0)][cols[0]], datasets[(datasets[cols[2]] == 0)][cols[1]],
                    marker='o', color='red', cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", label='无')
        plt.scatter(datasets[(datasets[cols[2]] == 1)][cols[0]], datasets[(datasets[cols[2]] == 0)][cols[1]],
                    marker='^', color='blue', cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", label='有')
        plt.title('是否购买割草机')
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.grid(True)
        plt.legend(loc='best')

# 1 数据加载
home_folder = os.path.expanduser("E:/pythonWp/pycharmWp/algorithm/")
data_folder = os.path.join(home_folder, "ML", "KNNStudy")
data_filename = os.path.join(data_folder, "KNNdatasets.csv")

knnClass=KNNClass()
dataset = knnClass.loadCsv(data_filename)
knnClass.drawOrginPic(dataset)

data_filename=os.path.join(data_folder, "KNNTestDatasets.csv")
testDatasets = knnClass.loadTestCsv(data_filename)
print(testDatasets)


for i in range(len(testDatasets)):
    print("------------%d-------------"%i)
    print(testDatasets[i:i+1])
    result = knnClass.main(data_filename, dataset, testDatasets[i:i+1],5)
    if result == 1:
        print("测试集属于第一类样本")
    else:
        print("测试集属于第二类样本")
    print("")
    knnClass.drawTestPic(dataset,testDatasets[i:i+1])







