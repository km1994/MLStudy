import math
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  algorithm.ML.RandomForest.xiechengRF.MyLog  import MyLog
random.seed(0)
#定义BPNeuralNetwork类， 使用三个列表维护输入层，隐含层和输出层神经元， 列表中的元素代表对应神经元当前的输出值.
# 使用两个二维列表以邻接矩阵的形式维护输入层与隐含层， 隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵.
class BPNeuralNetwork:
    def __init__(self,test_size=0.3,random_state=300):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []               #输入层神经元
        self.hidden_cells = []              #隐含层神经元
        self.output_cells = []              #输出层神经元
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.test_size=test_size        #测试样本比例
        self.random_state=random_state  #随机级别
        self.y_pred=[]                  #预测

    #产生随机数函数
    def rand(self,a, b):
        return (b - a) * random.random() + a

    # 创造一个指定大小的矩阵
    def make_matrix(self,m, n, fill=0.0):
        mat = []
        for i in range(m):
            mat.append([fill] * n)
        return mat

    # 定义sigmod函数
    def sigmoid(self,x):
        return 1.0 / (1.0 + math.exp(-x))

    # 定义sigmod函数的导数:
    def sigmoid_derivative(self,x):
        return x * (1 - x)

    # 1 划分数据集
    def divided(self, x_cases, y_labels):
        mylog.info("run divided function")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_cases, y_labels,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)

    # 2 定义setup方法初始化神经网络:
    def setup(self, ni, nh, no):
        mylog.info("run setup function")
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = self.make_matrix(self.input_n, self.hidden_n)
        self.output_weights = self.make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = self.rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = self.rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = self.make_matrix(self.input_n, self.hidden_n)
        self.output_correction = self.make_matrix(self.hidden_n, self.output_n)

    # 3 定义predict方法进行一次前馈， 并返回输出:
    def predict(self, inputs):
        mylog.info("run predict function")
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = self.sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = self.sigmoid(total)
        return self.output_cells[:]

    # 4 定义back_propagate方法定义一次反向传播和更新权值的过程， 并返回最终预测误差:
    def back_propagate(self, case, label, learn, correct):
        mylog.info("run back_propagate function")
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = self.sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = self.sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 5 定义train方法控制迭代， 该方法可以修改最大迭代次数， 学习率λ， 矫正率μ三个参数.
    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        mylog.info("run train function")
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)


    # 计算平均百分比偏误差 MAPD
    def calMAPE(self):
        mylog.info("run calMAPE function")
        i = 0
        MAPEerror = 0
        for y_test_item in self.y_test:
            MAPEerror = MAPEerror + (abs(y_test_item - self.y_pred[i]) / y_test_item)
            i = i + 1
        MAPE = (MAPEerror / len(self.y_test)) ** (1 / 2)
        print("MAPE:", MAPE)
        return MAPE

    # 计算变异系数 CV
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

    # 计算均方根误差 RMSE
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

    # 计算平均绝对偏差  MAD
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

    #编写test方法，演示如何使用神经网络学习异或逻辑:
    def main(self):
        mylog.info("run main function")
        xiechengDf=pd.read_csv('normalizedPCA_BP_Datadelete34.csv')
        xiecheng_cases_data = np.array(xiechengDf[['hotelClass','hotelLowestprice','surroundingsScore','serviceScore','facilityScore']])

        xiecheng_labels_data=np.array(xiechengDf['userRecommended']).reshape(len(xiechengDf['userRecommended']),1)

        #x_cases,y_labels,test_size=0.3
        self.divided(x_cases=xiecheng_cases_data,y_labels=xiecheng_labels_data)

        self.setup(5, 11, 1)
        self.train(self.x_train,self.y_train, 10000, 0.05, 0.1)

        self.y_pred = [self.predict(self.x_train[i]) for i in range(len(self.x_train))]

        for i in range(0,len(self.x_train)):
            print("---------------------------------------------------------------------")
            print("self.predict(xiecheng_cases_data[i]):", self.y_pred[i])
            print("xiecheng_labels_data[i]:", self.y_train[i])

        self.calMAD()
        self.calRMSE()
        self.calCV()
        self.calMAPE()





if __name__ == '__main__':
    mylog = MyLog()
    mylog.info("Start")
    nn = BPNeuralNetwork()
    nn.main()