import numpy as np
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

def kernelTrans(X, A, kTup):
    """
    核函数
    :param X: 全部向量
    :param A: 某个向量
    :param kTup: 核函数名称
    :return: :raise NameError:
    """
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T  #线性核函数
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A   #高斯径向基核函数
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))  #K依然是一个向量
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))                        #第一列是有效与否的标记
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

# 之后每次计算决策函数的时候就用这个软cache了：
def calcEk(oS, k):
    """
    计算第k个样本点的误差
    :param oS:
    :param k:
    :return:
    """
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

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
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:                                           #找不到，只好随机选择一个
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    '''
    当改变了模型参数后，通过如下方法更新eCache中的误差：
    :param oS:
    :param k:
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# eta和b的计算也要把内积换成核函数：
def innerL(i, oS):
    Ei = calcEk(oS, i)
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
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)                                                 # 更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])    #更新α_1
        updateEk(oS, i)                                                 # # 更新误差缓存
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

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


def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    """
    完整版的Platt SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :return:
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历间隔边界上的支持向量点
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # 翻转entireSet
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

import matplotlib.pyplot as plt
def plotSVM(dataArr, labelArr):
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

    plt.show()

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000,('rbf',k1))
    datMat=np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]                                                   #获取支持向量
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

    plotSVM(dataArr, labelArr)

#testRbf()




