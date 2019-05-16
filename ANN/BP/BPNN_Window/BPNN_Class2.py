# Back-Propagation Neural Networks
#
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>

# Adapted for instructional purposes by Bart Smeets <bartsmeets86@gmail.com>

import math
import random
import string
import matplotlib.pyplot as plt
import numpy as np
import time
from tkinter import *

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class BPNN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        self.h_weights = [1.0]*self.nh

        self.training_errors = []

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum) * self.h_weights[j]

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns,t):
        for p in patterns:
            test_info= "\n 测试结果： ",p[0], '->',self.update(p[0])
            print(test_info)
            t.insert(END,test_info)

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000,t='',N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        # 生成画布
        plt.figure(figsize=(8, 6), dpi=80)

        # 打开交互模式
        plt.ion()

        xList=[]
        errors=[]
        # 样本数量
        example_num = len(patterns)
        print("example_num: ",example_num)
        for i in range(iterations):
            # 清除原有图像
            plt.cla()
            # 设定标题等
            plt.title("动态曲线图")
            plt.grid(True)

            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

            xList.append(i)
            errors.append(error/example_num)
            plt.plot(xList, errors, "g-", linewidth=2.0, label="BPNN 误差变化曲线")
            # 暂停
            plt.pause(0.1)
            self.training_errors.append(error/example_num)

            if i % 10 == 0:
                train_info='\n iterations：%d'%iterations,',error %-.5f' % (error/example_num)
                print(train_info)
                t.insert(END, train_info)
                time.sleep(0.1)
                t.update()
                # 关闭交互模式
        plt.ioff()

        # 图形显示
        plt.show()

def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = BPNN(2, 4, 1)
    # train it with some patterns
    print("Training...")
    n.train(pat, iterations = 1000)
    print("Killing hnodes 1...")
    n.h_weights[0]=0
    print("Training...")
    n.train(pat, iterations = 1000)
    print("Killing hnodes 1 & 2...")
    n.h_weights[0]=0
    n.h_weights[1]=0
    print("Training...")
    n.train(pat, iterations = 1000)
    print("Killing hnodes 1 & 2  & 3...")
    n.h_weights[0]=0
    n.h_weights[1]=0
    n.h_weights[2]=0
    print("Training...")
    n.train(pat, iterations = 1000)
    print("Killing hnodes 1 & 2  & 3 & 4...")
    n.h_weights[0]=0
    n.h_weights[1]=0
    n.h_weights[2]=0
    n.h_weights[3]=0
    print("Training...")
    n.train(pat, iterations = 1000)

    # test it
    print("Testing...")
    n.test(pat)



    with open( 'error_results.txt', 'w' ) as f:
        for v in n.training_errors:
            f.write(str(v) + '\n')


if __name__ == '__main__':
    demo()