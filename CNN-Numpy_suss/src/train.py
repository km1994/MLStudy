#coding:utf-8
import numpy as np
import scipy.io as scio
import sys
sys.path.append('../')
from convnet import cnn
import config
import signal
import sys
import time
from Tkinter import *
import matplotlib
import matplotlib.pyplot as plt


numEpoch = config.numEpoch
trainExamples = config.trainExamples
valExamples = config.valExamples
batchSize = config.batchSize
saveModel = config.saveModel
modelFile = config.modelFile
validate  = config.validate
pretrain = config.pretrain
trainedModel = config.trainedModel

log = config.log
trainlog = config.trainlog
vallog = config.vallog

net = cnn()

if pretrain:
    model = scio.loadmat(trainedModel)
    net.Weights = np.asarray(model['weights'][0])
    for i in range(5):
        net.Biases[i] = model['biases'][0][i][0]

mnist = scio.loadmat('data/mnist_2D.mat')

def signal_handler(signal, frame):
            print('You pressed Ctrl+C!')
            if saveModel:
                SaveModel()
                print("Model Saved")
            sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def SaveModel():
    if saveModel:
        model= {}
        model['weights'] = net.Weights
        model['biases'] = net.Biases
        scio.savemat(modelFile, model)

def train(END,t):
    print("-----------train------------")
    t.insert(END, "-----------train------------")
    # 生成画布
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式
    plt.ion()
    
    if log:
        trlog = open( trainlog , 'w')
        vlog = open( vallog, 'w')
    numIter = 1
    x_list=[]
    val_loss_list=[]
    for epoch in range(numEpoch):
        trainList = [np.random.randint(0,60000) for i in range(trainExamples)]
        valList = [np.random.randint(0,10000) for i in range(valExamples)]

        trainLabel = np.asarray([[0 for i in range(10)] for j in range(trainExamples)])
        trainData = np.zeros(( trainExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))

        valLabel = np.asarray([[0 for i in range(10)] for j in range(valExamples)])
        valData = np.zeros(( valExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))

        j=0
        for i in trainList:
            trainLabel[j,mnist['Y_train'][i]] = 1
            trainData[j] = mnist['X_train'][i]
            j += 1

        j=0
        for i in valList:
            valLabel[j,mnist['Y_test'][i]] = 1
            valData[j] = mnist['X_test'][i]
            j += 1


        j = 0
        while( j < trainExamples ):

            batchData = trainData[j:j+batchSize]
            batchLabel = trainLabel[j:j+batchSize]

            batchLoss = net.backward(batchData, batchLabel)

            train_process_info = '\n Iteration {0} : Train Loss = {1} '.format(str(numIter),str(batchLoss))
            t.insert(END, train_process_info)
            time.sleep(0.1)
            t.update()
            print(train_process_info)
            

            if log:
                trlog.write(str(numIter) + ' ' + str(batchLoss) + '\n')

            numIter += 1
            j += batchSize

        ### Validation
        acc = 0
        val_loss = 0
        for i in range(valExamples):
            [predict,loss] = net.validate(valData[i], valLabel[i])

            if valLabel[i][predict] == 1:
                acc += 1

            val_loss += loss

        if epoch % 10 ==0:
            # 清除原有图像
            plt.cla()

            # 设定标题等
            #plt.set_title("动态曲线图")
            plt.grid(True)
            x_list.append(epoch)
            val_loss_list.append(val_loss/valExamples)
            plt.plot(x_list,val_loss_list, "b--", linewidth=2.0)
            plt.pause(0.1)

        if validate:
            train_accuracy_info = '\n Epoch {0} : Validation Loss: {1} ,Accuracy: {2} '.format(str(epoch+1),str(val_loss/valExamples),str(acc*100.0/valExamples))
            print(train_accuracy_info)
            t.insert(END, train_accuracy_info)
            time.sleep(0.1)
            t.update()
          

        if log:
            vlog.write(str(epoch+1)+ '  '+ str(val_loss/valExamples)+'  '+ str(acc*100.0/valExamples)+ '\n')

    # 关闭交互模式
    plt.ioff()

    # 图形显示
    plt.show()

    if log:
        trlog.close()
        vlog.close()

    SaveModel()
    print("模型保存完毕！")
    


