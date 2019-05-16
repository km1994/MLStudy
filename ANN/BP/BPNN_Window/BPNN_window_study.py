# coding:utf-8
from tkinter import *
import time
import sys
from  BPNN_Window import Load_Data_tools as Load_Data_tools
from BPNN_Window import BPNN_Class2 as bpnn
import tkinter.filedialog as filedialog
import numpy as np


sys.path.append("..")

root = Tk()
root.title("BPNN 卷积神经网络")
dataMat=[]

path=''
def selectPath(var,t):
    global path
    global dataMat
    text = '载入样本'
    path_ = filedialog.askopenfilename(title='选择文件')#选择文件
    t.insert(END, "文件地址：%s"%path_)
    path=path_
    print("path: ",path)
    dataMat = Load_Data_tools.switch_load_data_fun(2, path)
    print("dataMat:", dataMat)
    var.set(text)


def loadDataFun(var):
    global dataMat
    text = '预存样本载入完成'
    dataMat = Load_Data_tools.switch_load_data_fun(1, "")
    print("dataMat:", dataMat)
    var.set(text)

def randDataFun(t):
    global dataMat
    print("------随机生成训练数据--------")
    dataMat_num=10
    for i in range(0,dataMat_num):
        data_list = []
        train_data=np.random.rand(2).tolist()
        data_list.append(train_data)
        train_label = np.random.rand(1).tolist()
        data_list.append(train_label)
        dataMat.append(data_list)
    print(dataMat)
    rand_data="随机生成数据：",str(dataMat)
    t.insert(END,rand_data)


bpnn_model=None
def trainFun(status_var,t):
    global bpnn_model
    text = '样本训练完成'
    bpnn_model = bpnn.BPNN(2, 40, 1)
    # train it with some patterns
    bpnn_model.train(dataMat,100,t)
    print("dataMat:", dataMat)
    status_var.set(text)

testMat=[
    [[10.2351, 14.3846], [0]], [[8.42297, 10.9991], [0]], [[7.62384, 10.6236], [0]]
    ]
def testFun(status_var,t):
    global bpnn_model
    global testMat
    text = '测试样本完成'
    bpnn_model.test(testMat,t)
    status_var.set(text)


def BPNNWindow(master):
    master.geometry('550x550+50+50')
    frame = Frame(master)
    frame.pack()
    status_var = StringVar()
    t = Text()
    t.pack()

    Button(frame, text="载入预存样本", font=('Verdana', 15), command=lambda: loadDataFun(status_var)).grid(row=0, column=0)
    Button(frame, text="载入样本", font=('Verdana', 15), command=lambda: selectPath(status_var,t)).grid(row=0, column=1)
    Button(frame, text="随机生成", font=('Verdana', 15), command=lambda: randDataFun(t)).grid(row=0, column=2)
    Button(frame, text="训练样本", font=('Verdana', 15), command=lambda: trainFun(status_var,t)).grid(row=1, column=1)

    Button(frame, text="测试样本", font=('Verdana', 15), command=lambda: testFun(status_var,t)).grid(row=2, column=1)
    Label(frame, text='状态', font=('Verdana', 15), textvariable=status_var).grid()


app = BPNNWindow(root)
print(dataMat)
root.mainloop()