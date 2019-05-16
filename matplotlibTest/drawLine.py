"""
=======================================
A simple plot with a custom dashed line
=======================================

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""
import numpy as np
import matplotlib.pyplot as plt

class drawLineClass:
    # 初始化函数
    def __init__(self,type=0):
        self.x=[]
        self.y=[]
        self.xLabel="xLabel"
        self.yLabel="yLabel"
        self.linewidth=1
        self.lineType='-'
        self.legendLabel="legendLabel"
        self.lineColor="green"
        if type ==1:
            self.fig, self.ax=self.createSubplots()           #创建画板


    #传入x
    def setX(self,x):
        self.x=x

    # 传出x
    def getX(self):
        return self.x

    # 传入x
    def setY(self, y):
        self.y = y

    # 传出x
    def getY(self):
        return self.y

    # 传入xLable
    def setXLabel(self, xLabel):
        self.xLabel = xLabel
        self.ax.set_xlabel(self.xLabel)

    # 传入yLable
    def setYLabel(self,yLabel):
        self.yLabel = yLabel
        self.ax.set_ylabel(self.yLabel)

            #创建画板

    def setlegendLabel(self,legendLabel):
        self.legendLabel=legendLabel

    def createSubplots(self):
        fig, ax = plt.subplots()
        return fig,ax

    #设置线宽度
    def setLinewidth(self,linewidth):
        self.linewidth=linewidth

    #设置线样式
    def setLineType(self,lineType):
        self.lineType=lineType

    #设置线颜色
    def setLineColor(self,lineColor):
        self.lineColor=lineColor

    #划线
    def drawLine(self):
        if self.lineType=='':
            self.lineType='--'
        line1, = self.ax.plot(self.x, self.y, self.lineType,color=self.lineColor, linewidth=self.linewidth,label=self.legendLabel)
        self.lineType = ''

        self.ax.legend(loc='best')

    #显示
    def show(self):
        plt.show()

if __name__ == '__main__':
    x=[1,2,4,5,6,6]
    y=[2,4,5,6,7,3]
    drawLine=drawLineClass(0)
    drawLine.setXLabel("x")
    drawLine.setYLabel("y")

    drawLine.setX(x)
    drawLine.setY(y)
    drawLine.setLineColor("red")
    drawLine.setlegendLabel("第一条曲线")
    drawLine.setLineType("--")
    drawLine.drawLine()

    x = [1, 3, 4, 6, 6, 6]
    y = [2, 5, 4, 8, 7, 7]
    drawLine.setX(x)
    drawLine.setY(y)
    drawLine.setLineColor("green")
    drawLine.setlegendLabel("第二条曲线")
    drawLine.setLineType("-.")
    drawLine.drawLine()

    drawLine.show()
