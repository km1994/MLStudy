#coding:utf-8
from Tkinter import *
import time
import sys
sys.path.append("..")
from data.make2D import originPrepare
from src.train import train
from src.test import test



root = Tk()
root.title("CNN 卷积神经网络")

def callback(n,var):
    text=''
    if n == 1:
        text='原始样本载入完成'
        originPrepare()
    elif n == 2:
        text='载入样本'
    elif n == 3:
        text='随机生成'
    elif n == 5:
        text='测试样本'
        
    var.set(text)
    print n

def loadDataFun(var):
    text='原始样本载入完成'
    originPrepare()
    var.set(text)

def trainFun(status_var,t):
	text='样本训练完成'
	status_var.set(text)
	train(END,t)

def testFun(status_var,t):
	text='测试样本'
	status_var.set(text)
	test(END,t)

class App:
	def __init__(self, master):
		master.geometry('550x550+50+50')
		frame = Frame(master)
		frame.pack()

		
		status_var = StringVar()
		
		t = Text()
		t.pack()

		Button(frame, text="原始样本",font=('Verdana', 15),command=lambda: loadDataFun(status_var) ).grid(row=0,column=0)
		Button(frame, text="载入样本",font=('Verdana', 15),command=lambda: callback(2,status_var) ).grid(row=0,column=1)
		Button(frame, text="随机生成",font=('Verdana', 15),command=lambda: callback(3,status_var) ).grid(row=0,column=2)
		Button(frame, text="训练样本",font=('Verdana', 15),command=lambda: trainFun(status_var,t) ).grid(row=1,column=1)
		Button(frame, text="测试样本",font=('Verdana', 15),command=lambda: testFun(status_var,t) ).grid(row=2,column=1)
		Label(frame,text = '状态',font=('Verdana', 15),textvariable=status_var).grid()
		


app = App(root)
root.mainloop()
