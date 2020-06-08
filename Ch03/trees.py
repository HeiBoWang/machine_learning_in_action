

# coding=utf-8
# -*- coding: utf-8 -*-

'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

# 初始化数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

"""
首先， 计算数据集中实例的总数。 我们也可以在需要时再计算这个值， 但是由于代码中多次用到这个值， 为了
提高代码效率， 我们显式地声明一个变量保存实例总数。 然后， 创建一个数据字典， 它的键值是最后一列的数值❶。 如果当前键值不存在， 则
扩展字典并将当前键值加入字典。 每个键值都记录了当前类别出现的次数。 最后， 使用所有类标签的发生频率计算类别出现的概率。 我们将用
这个概率计算香农熵❷， 统计所有类标签发生的次数。
"""
# 此代码的功能是计算给定数据集的熵。
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 1. （以下五行） 为所有可能分类创建字典
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # 以2为底求对数
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

"""
上节我们学习了如何度量数据集的无序程度， 分类算法除了需要测量信
息熵， 还需要划分数据集， 度量划分数据集的熵， 以便判断当前是否正
确地划分了数据集。 我们将对每个特征划分数据集的结果计算一次信息
熵， 然后判断按照哪个特征划分数据集是最好的划分方式。 想象一个分
布在二维空间的数据散点图， 需要在数据之间划条线， 将它们分成两部
分， 我们应该按照x轴还是y轴划线呢？
------------------------------------------------
代码使用了三个输入参数： 待划分的数据集、 划分数据
集的特征、 需要返回的特征的值。 需要注意的是， Python语言不用考虑
内存分配问题。 Python语言在函数中传递的是列表的引用， 在函数内部
对列表对象的修改， 将会影响该列表对象的整个生存周期。 为了消除这
个不良影响， 我们需要在函数的开始声明一个新列表对象。 因为该函数
代码在同一数据集上被调用多次， 为了不修改原始数据集， 创建一个新
的列表对象❶。 数据集这个列表中的各个元素也是列表， 我们要遍历数
据集中的每个元素， 一旦发现符合要求的值， 则将其添加到新创建的列
表中。 在if语句中， 程序将符合特征的数据抽取出来❷。 后面讲述得更
简单， 这里我们可以这样理解这段代码： 当我们按照某个特征划分数据
集时， 就需要将所有符合要求的元素抽取出来。 代码中使用了Python语
言list类型自带的extend()和append()方法。 这两个方法功能类似
"""
# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    # 1. 创建新的list对象
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 2. （以下三行） 抽取
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
接下来我们将遍历整个数据集， 循环计算香农熵和splitDataSet()函
数， 找到最好的特征划分方式。 熵计算将会告诉我们如何划分数据集是
最好的数据组织方式
------------------------------
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        # 1. （ 以下两行） 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        # 2. （ 以下五行） 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            # 3. 计算最好的信息增益
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

"""
第一个结束条件使得算法可以终止， 我们甚至可以设置算法可以划分的
最大分组数目。 后续章节还会介绍其他决策树算法， 如C4.5和CART，
这些算法在运行时并不总是在每次划分分组时都会消耗特征。 由于特征
数目并不是在每次划分数据分组时都减少， 因此这些算法在实际使用时
可能引起一定的问题。 目前我们并不需要考虑这个问题， 只需要在算法
开始运行前计算列的数目， 查看算法是否使用了所有属性即可。 如果数
据集已经处理了所有属性， 但是类标签依然不是唯一的， 此时我们需要
决定如何定义该叶子节点， 在这种情况下， 我们通常会采用多数表决的
方法决定该叶子节点的分类
-------------------
该函数使用分类名称的列表， 然后创建键值为classList中唯一值的数据字
典， 字典对象存储了classList中每个类标签出现的频率， 最后利用
operator操作键值排序字典， 并返回出现次数最多的分类名称。
"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
使用两个输入参数： 数据集和标签列表。 标签列表
包含了数据集中所有特征的标签， 算法本身并不需要这个变量， 但是为了给出数据明确的含义， 我们将它作为一个输入参数提供。 此外， 前面
提到的对数据集的要求这里依然需要满足。 上述代码首先创建了名
为classList的列表变量， 其中包含了数据集的所有类标签。 递归函数
的第一个停止条件是所有的类标签完全相同， 则直接返回该类标签❶。
递归函数的第二个停止条件是使用完了所有特征， 仍然不能将数据集划
分成仅包含唯一类别的分组❷。 由于第二个条件无法简单地返回唯一的
类标签， 这里使用程序清单3-3的函数挑选出现次数最多的类别作为返回值
"""
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 1.  （以下两行） 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 2. （以下两行） 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 3. 得到列表包含的所胡属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
