# -*- coding: utf-8 -*-
import numpy as np
import os
#from common_libs import *
import matplotlib.pyplot as plt

###################################
#加载数据文件并转化为矩阵
#input:path(str):数据文件路径
#input:delimiter(str):文件分隔符
#output:recordlist(mat):转换后的矩阵形式
###################################
def file2matrix(path,delimiter):
    print("-----------loadTxtData-----------")
    recordlist = []
    fr = open(path)  # 打开文件
    for line in fr.readlines():
        lines = line.strip().split(delimiter)
        lineArr = []
        for i in range(len(lines)):
            lineArr.append(lines[i])
        recordlist.append(lineArr)
    fr.close()
    # return dataMat, labelMat
    # fp=open(path,"r")
    # content=fp.read()
    # fp.close()
    # rowlist=content.splitlines()    #按行转换为一维表
    # #逐行遍历，结果按分隔符分割为行向量
    # recordlist=[map(eval,row.split(delimiter)) for row in rowlist if row.strip()]
    return np.mat(recordlist)          #返回转换后的矩阵形式

###################################
#绘制分类点
#input:plt(str):绘图函数包
#input:input(str):数据集
###################################
def drawScatterbyLabel(plt,input):
    m,n=np.shape(input)
    target=input[:,-1]
    print("--------------",target)
    for i in range(m):
        if target[i] == 0:
            plt.scatter(input[i,0],input[i,1],
                        marker='o', color='red', cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", label='无')
            #plt.scalar(input[i,0],input[i,1],c='blue',marker="o")
        else:
            plt.scatter(input[i, 0], input[i, 1],
                        marker='.', color='blue', cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="red", label='a')

###################################
# 构建B+X矩阵，默认B为全1 的列向量
# input:dataSet(mat):数据集
# outpur:dataMat(mat):数据集
###################################
def buildMat(dataSet):
    m,n=np.shape(dataSet)
    dataMat=np.zeros((m,n))
    dataMat[:,0]=1
    dataMat[:,1:]=dataSet[:,:-1]
    return dataMat

###################################
#Logisitic 函数
#input:wTx(str):数据集
#output:logistic(float):logistic函数计算结果
###################################
def logistic(wTx):
    return 1.0/(1.0+np.exp(-wTx))

#1 导入数据
input=file2matrix("testSet.txt","\t")
#print(input)
label=input[:,-1].astype('float64')
[m,n]=np.shape(input)

#2 构建b+x 系数矩阵；b默认为1
dataMat=np.mat(buildMat(input))


#3 定义步长和迭代次数
alpha=0.001     #步长
steps=500       #迭代次数

weights=np.ones((n,1))      #初始化权重向量
errorlist=[]

#4 主程序
for k in range(steps):
    net=dataMat*np.mat(weights)             #待估计网络 net(100,1)=dataMat(100,3)*weight(3,1)
    output=logistic(net)                    #logistic函数 output(100,1)
    loss=np.array(output-label)             #loss(100,1)=output(100,1)*label(100,1)
    error=0.5*np.sum(loss*loss)             #loss 函数 error(1,1)=loss(100,1)*loss(100,1)
    errorlist.append(error)
    grad=dataMat.T*loss                     #梯度 grad(3,1)=dataMat.T(3,100)*loss(100,1)
    weights=weights-alpha*grad              #迭代 weights(3,1)=weights(3,1)-alpha

print(weights)      #输出训练后的权重
drawScatterbyLabel(plt,input)
X=np.linspace(-5,5,100)
Y=(np.double(weights[0]+X*(np.double(weights[1]))))/np.double(weights[2])
print("---------",(X,Y))
plt.plot(X,Y)
plt.show()















