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

def test(status_var,t):
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


    model = scio.loadmat("./models/1_0.01_model.mat")
    net.Weights = np.asarray(model['weights'][0])
    for i in range(5):
        net.Biases[i] = model['biases'][0][i][0]

    mnist = scio.loadmat('./data/mnist_2D.mat')


    trainList = [np.random.randint(0,60000) for i in range(trainExamples)]
    valList = [np.random.randint(0,10000) for i in range(valExamples)]


    valLabel = np.asarray([[0 for i in range(10)] for j in range(valExamples)])
    valData = np.zeros(( valExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))


    j=0
    for i in valList:
        valLabel[j,mnist['Y_test'][i]] = 1
        valData[j] = mnist['X_test'][i]
        j += 1



    ### Validation
    acc = 0
    val_loss = 0
    for i in range(valExamples):
        [predict,loss] = net.validate(valData[i], valLabel[i])

        if valLabel[i][predict] == 1:
            acc += 1

        val_loss += loss


    if validate:
        test_accuracy_info = '\n Validation Loss: {0} ,Accuracy: {1} '.format(str(val_loss/valExamples),str(acc*100.0/valExamples))
        print(test_accuracy_info)
        t.insert(END, test_accuracy_info)
    



