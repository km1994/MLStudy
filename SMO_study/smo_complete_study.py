import numpy as np
class optStructK:
    '''
        为了避免频繁填充函数参数，定义了一个中间结构：
    '''
    def __init__(self,dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))

def loadDataSet(fileName):
    """
    加载数据集
    input:
        fileName: string
    output:
        dataMat: Mat
        labelMat:Mat
    """
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def calcEkK(oS, k):
    """
    计算第k个样本点的误差
    :param oS:
    :param k:
    :return:
    """
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def updateEkK(oS, k):
    '''
    当改变了模型参数后，通过如下方法更新eCache中的误差：
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEkK(oS, k)
    oS.eCache[k] = [1,Ek]

def selectJrand(i, m):
    """
    随机从0到m挑选一个不等于i的数
    :param i:
    :param m:
    :return:
    """
    j = i  # 排除i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

def selectJK(i, oS, Ei):
    """
    选择第2个变量的过程，即内层循环中的启发规则。选择标准是使alpha_2有足够大的变化。
    :param i:第一个变量
    :param oS:中间结果
    :param Ei:第i个点的误差
    :return:
    """
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]                           #设为有效
    # 找寻产生最大误差变化的alpha
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]  #有效位为1的误差列表
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:                   #遍历列表找出最大误差变化
            if k == i: continue                     #第二个alpha不应该等于第一个
            Ek = calcEkK(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:                                           #找不到，只好随机选择一个
        j = selectJrand(i, oS.m)
        Ej = calcEkK(oS, j)
    return j, Ej

def clipAlpha(aj, H, L):
    """
    将aj剪裁到L(ow)和H(igh)之间
    :param aj:
    :param H:
    :param L:
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def innerLK(i, oS):
    Ei = calcEkK(oS, i)
    #如果误差太大，且alpha满足约束，则尝试优化它
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJK(i, oS, Ei)   #不再是 selectJrand 那种简化的选择方法
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy() # 教材中的α_1^old和α_2^old
        if (oS.labelMat[i] != oS.labelMat[j]):                           # 两者所在的对角线段端点的界
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEkK(oS, j)                                                 # 更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])    #更新α_1
        updateEkK(oS, i)                                                 # # 更新误差缓存
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整版的Platt SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :return:
    """
    oS = optStructK(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerLK(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历间隔边界上的支持向量点
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerLK(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # 翻转entireSet
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

import matplotlib.pyplot as plt
def plotSVM(dataArr, labelArr, w, b, svList):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers = []
    colors = []
    for i in range(len(labelArr)):
        xPt = dataArr[i][0]
        yPt = dataArr[i][1]
        label = labelArr[i]
        if (label == -1):
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:
            xcord1.append(xPt)
            ycord1.append(yPt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0, ycord0, marker='s', s=90)
    ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')
    for sv in svList:
        circle = plt.Circle((dataArr[sv][0], dataArr[sv][1]), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,
                        alpha=0.5)
        ax.add_patch(circle)

    w0 = w[0][0]
    w1 = w[1][0]
    b = float(b)
    x = np.arange(-2.0, 12.0, 0.1)
    y = (-w0 * x - b) / w1
    ax.plot(x, y)
    ax.axis([-2, 12, -8, 6])
    plt.show()

def calcWs(alphas,dataArr,classLabels):
    """
    根据支持向量计算分离超平面(w,b)的w参数
    :param alphas:拉格朗日乘子向量
    :param dataArr:数据集x
    :param classLabels:数据集y
    :return: w=∑alphas_i*y_i*x_i
    """
    X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def main():
    dataArr, labelArr = loadDataSet('testSet.txt')
    b, alphas = smoPK(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    w = calcWs(alphas, dataArr, labelArr)
    print(w)
    svList = []
    for i in range(len(alphas)):
        if abs(alphas[i]) > 0.0000001:
            print(dataArr[i], labelArr[i])
            svList.append(i)

    plotSVM(dataArr, labelArr, w, b, svList)




