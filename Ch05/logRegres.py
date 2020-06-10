
# coding=utf-8

from numpy import *


"""
准备数据
主要功能是打开文本文件testSet.txt并逐行读取。 每行前两个值分别是X1和X2， 第三个值是数据对应的类别标签。 
此外， 为了方便计算，该函数还将X0的值设为1.0
"""


def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

"""
sigmoid 函数实现
"""
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

"""
梯度下降算法的实现
该函数有两个参数。 第一个参数是dataMatIn， 它是一个2维NumPy数组， 每列分别代表每个不同的特征， 每行则代表每个训练样本。 我们现在采用的
是100个样本的简单数据集， 它包含了两个特征X1和X2， 再加上第0维特征X0， 所以dataMath里存放的将是100×3的矩阵。 在❶处， 我们获得
输入数据并将它们转换成NumPy矩阵。
第二个参数是类别标签， 它是一个1×100的行向量。 为了便于矩阵运算， 需要将该行向量转换为列向量， 做法是将原向
量转置， 再将它赋值给labelMat。 接下来的代码是得到矩阵大小， 再设置一些梯度上升算法所需的参数。
变量alpha是向目标移动的步长， maxCycles是迭代次数。 在for循环迭代完成后， 将返回训练好的回归系数。 需要强调的是， 在❷处的运算
是矩阵运算。 变量h不是一个数而是一个列向量， 列向量的元素个数等于样本个数， 这里是100。 对应地， 运算dataMatrix * weights代表
的不是一次乘积计算， 事实上该运算包含了300次的乘积
"""
def gradAscent(dataMatIn, classLabels):
    # 1. （以下两行） 转换为NumPy矩阵数据类型
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        # todo 逻辑回归算法核心
        # 2. （以下三行） 矩阵相乘
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

# --------------------让我们测试一下梯度下降算法----------------------------
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
# [[ 4.12414349]
#  [ 0.48007329]
#  [-0.6168482 ]]
# print weights
print weights.getA()

# ----------------------分析数据： 画出决策边界 -----------------------------
"""
画出数据集和logistic回归最佳拟合直线的函数

代码是直接用Matplotlib画出来的。 唯一要指出的是，❶处设置了sigmoid函数为0。 回忆5.2节， 0是两个类别（类别1和类别
0） 的分界处。 因此， 我们设定 0 = w0x0 + w1x1 + w2x2， 然后解出X2和X1的关系式（即分隔线的方程， 注意X0=1） 。
"""
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()



# ---------------------------画出边界-------------------------------
# getA()函数与mat()函数的功能相反，是将一个numpy矩阵转换为数组
plotBestFit(weights.getA())

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

"""
改进的随机梯度上升算法
第一处改进在❶处,一方面， alpha在每次迭代的时候都会调整， 这会缓解图5-6上的数据波动或者高频波动。 另外， 虽然alpha会随着迭代次数
不断减小， 但永远不会减小到0， 这是因为❶中还存在一个常数项。 必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影
响。 如果要处理的问题是动态变化的， 那么可以适当加大上述常数项，来确保新的值获得更大的回归系数。
第二个改进的地方在❷处， 这里通过随机选取样本来更新回归系数。 这种方法将减少周期性的波动（如图5-6中的波动） 。 具体
实现方法与第3章类似， 这种方法每次随机从列表中选出一个值， 然后从列表中删掉该值（再进行下一次迭代） 
"""


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # 1. alpha每次迭代时需要调整
            alpha = 4/(1.0+j+i)+0.0001
            # 2. 随机选取更新
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# --------------------------------利用该分类器来预测病马的生死问题。 ----------------------------------


"""
回归系数的更新公式如下:weights = weights + alpha * error * dataMatrix[randIndex,如果dataMatrix的某特征对应值为0,由于sigmoid(0)=0.5， 即它对结果的预测不具有任何倾向性
classifyVector它以回归系数和特征向量作为输入来计算对应的Sigmoid值。 如果Sigmoid值大于0.5函数返回1， 否则返回0。
"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


"""
colicTest()， 是用于打开测试集和训练集， 并对数据进行格式化处理的函数。 该函数首先导入训练集， 同前面一样， 数据的
最后一列仍然是类别标签。 数据最初有三个类别标签， 分别代表马的三种情况： “仍存活”、 “已经死亡”和“已经安乐死”。 这里为了方便， 将“已
经死亡”和“已经安乐死”合并成“未能存活”这个标签 。 数据导入之后，便可以使用函数stocGradAscent1()来计算回归系数向量。 这里可以
自由设定迭代的次数， 例如在训练集上使用500次迭代， 实验结果表明这比默认迭代150次的效果更好。 在系数计算完成之后， 导入测试集并
计算分类错误率。 整体看来， colicTest()具有完全独立的功能
"""
def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate


"""
multiTest()， 其功能是调用函数colicTest()10次并求结果的平均值
"""


def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

# 完整功能的测试
multiTest()
