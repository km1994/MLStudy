# encoding = utf8
'''
    @Author: King
    @Date: 2019.05.16
    @Purpose: 机器学习算法学习与实现
    @Introduction:  A机器学习算法学习与实现
    @Datasets: 
    @Link : 
    @Reference : 
'''
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from  algorithm.ML.RandomForest.xiechengRF.MyLog  import MyLog
from sklearn.ensemble import RandomForestClassifier
from algorithm.ML.matplotlibTest.drawLine import drawLineClass

class RandomForestClass:
    #初始化函数
    def __init__(self,data,test_size=0.3,random_state=300,n_estimators=10,max_depth=5):
        self.data=data
        self.data=self.pretreatment()           #缺失值处理
        #self.print()
        self.X=[]
        self.y=[]
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        self.test_size=test_size                #指定参数test_size=0.3,数据样本作为测试集的比例为30%，输出训练集和测试集大小
        self.random_state=random_state          #随机发生器If int, random_state is the seed used by the random number generator;
                                                # If RandomState instance, random_state is the random number generator;
                                                # If None, the random number generator is the RandomState instance used by np.random.
        self.divided()

        self.y_pred=[]
        self.model=self.RandomForestMain(n_estimators=n_estimators,max_depth=max_depth)    #其构建随机森林分类模型，指定n_estimators参数为10，
                                                                        # 即使用10棵决策树构建模型。将训练集传入模型进行模型训练。
        self.predict()

        self.MAPE=self.calMAPE()
        self.CV=self.calCV()
        self.RMSE=self.calRMSE()
        self.MAD=self.calMAD()

    #1 预处理
    def pretreatment(self):
        mylog.info("run pretreatment function")
        data =  self.data.fillna(self.data.mean())
        #self.print(data=data,type=0)
        return data

    #2 划分数据集
    def divided(self):
        mylog.info("run divided function")
        self.X=self.data.iloc[:,:6]
        self.y = self.data.iloc[:,6]

        self.print(type=0,data=self.X)
        self.print(type=0, data=self.y)
        #print("self.test_size",self.test_size)
        #print("self.random_state", self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)


    #3 构建随机森林模型并训练
    def RandomForestMain(self,n_estimators=10,max_depth=5):
        mylog.info("run RandomForestMain function")
        model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        model.fit(self.X_train, self.y_train)
        return model


    #4 利用随机森林模型预测分类
    def predict(self):
        mylog.info("run predict function")
        self.y_pred = self.model.predict(self.X_test)
        #print("Predictions of test set:\n%s" % self.y_pred)

    #计算平均百分比偏误差 MAPD
    def calMAPE(self):
        mylog.info("run calMAPE function")
        i = 0
        MAPEerror = 0
        for y_test_item in self.y_test:
            MAPEerror = MAPEerror + (abs(y_test_item - self.y_pred[i]) / y_test_item)
            i = i + 1
        MAPE = (MAPEerror / len(self.y_test)) ** (1 / 2)
        print("MAPE:",MAPE)
        return MAPE

    #计算变异系数 CV
    def calCV(self):
        mylog.info("run calCV function")
        i = 0
        CVerrors = 0
        for y_test_item in self.y_test:
            CVerrors = CVerrors + (y_test_item - self.y_pred[i]) ** 2
            i = i + 1
        CV = (CVerrors / np.mean(self.y_test))
        print("CV:", CV)
        return CV

    #计算均方根误差 RMSE
    def calRMSE(self):
        mylog.info("run calRMSE function")
        i = 0
        RMSEerrors = 0
        for y_test_item in self.y_test:
            RMSEerrors = RMSEerrors + (y_test_item - self.y_pred[i]) ** 2
            i = i + 1
        RMSE = (RMSEerrors / len(self.y_test)) ** (1 / 2)
        print("RMSE:", RMSE)
        return RMSE

    #计算平均绝对偏差  MAD
    def calMAD(self):
        mylog.info("run calMAD function")
        i = 0
        MADerror = 0
        for y_test_item in self.y_test:
            MADerror = MADerror + abs(self.y_pred[i] - y_test_item)
            i = i + 1
        MAD = MADerror / len(self.y_test)
        print("MAD:", MAD)
        return MAD

    #打印变量
    def print(self,data="",type=1):
        if type ==1:
            print(self.data)
        else:
            print(data)


if __name__ == '__main__':
    data = pd.read_csv("normalizedPCARFDatadelete4.csv", index_col=0)
    mylog = MyLog()
    mylog.info("Start")
    MAPEarr=[]
    CVarr=[]
    RMSEarr=[]
    MADarr=[]
    for i in range(1,16):
        rf = RandomForestClass(data,0.3,300,i,4)
        MAPEarr.append(rf.calMAPE())
        CVarr.append(rf.calCV())
        RMSEarr.append(rf.calRMSE())
        MADarr.append(rf.calMAD())

    print("MAPEarr:",MAPEarr)
    print("CVarr:", CVarr)
    print("RMSEarr:", RMSEarr)
    print("MADarr:", MADarr)

    x=range(1,len(MAPEarr)+1)
    drawLine=drawLineClass(0)

    #画 平均百分比偏误差 MAPD
    drawLine.fig, drawLine.ax=drawLine.createSubplots()
    drawLine.setX(x)
    drawLine.setY(MAPEarr)
    drawLine.setLineColor("green")
    drawLine.setlegendLabel("平均百分比偏误差 MAPD")
    drawLine.setLineType("-.")
    drawLine.drawLine()

    # 画 计算变异系数 CV
    drawLine.fig, drawLine.ax =drawLine.createSubplots()
    drawLine.setY(CVarr)
    drawLine.setLineColor("red")
    drawLine.setlegendLabel("计算变异系数 CV")
    drawLine.setLineType("--")
    drawLine.drawLine()

    # 画 计算均方根误差 RMSE
    drawLine.fig, drawLine.ax =drawLine.createSubplots()
    drawLine.setY(RMSEarr)
    drawLine.setLineColor("yellow")
    drawLine.setlegendLabel("计算均方根误差 RMSE")
    drawLine.setLineType("*-")
    drawLine.drawLine()

    # 画 计算平均绝对偏差  MAD
    drawLine.fig, drawLine.ax =drawLine.createSubplots()
    drawLine.setY(MADarr)
    drawLine.setLineColor("orange")
    drawLine.setlegendLabel("计算平均绝对偏差  MAD")
    drawLine.setLineType("+-")
    drawLine.drawLine()

    drawLine.show()




