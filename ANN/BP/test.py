#coding:utf-8
# from algorithm.ML.ANN.BP.BP import *

from BP import *
#数据集
bpnet = BPNet()
bpnet.loadDataSet("testSet2.txt")
bpnet.dataMat = bpnet.normlize(bpnet.dataMat)

#绘制数据集的散点图
bpnet.drawClassScatter(plt)

#BP神经网络进行数据分类
bpnet.bpTrain()
print(bpnet.out_wb)
print(bpnet.hi_wb)

#计算和绘制分类线
x,z = bpnet.BPClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
plt.show()

#绘制误差线
bpnet.TrenLine(plt)
plt.show()