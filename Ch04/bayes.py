
# coding=utf-8

from numpy import *

""" 1
函数loadDataSet()创建了一些实验样本。 该函数返回的第一个变量是进行词条切分后的文档集合， 这些文档来自斑点犬爱好者留言
板。 这些留言文本被切分成一系列的词条集合， 标点符号从文本中去掉， 
loadDataSet( )函数返回的第二个
变量是一个类别标签的集合。 这里有两类， 侮辱性和非侮辱性。 这些文本的类别由人工标注， 这些标注信息用于训练程序以便自动检测侮辱性留言
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

""" 2 
函数createVocabList()会创建一个包含在所有文档中出现的不重复词的列表， 为此使用了Python的set数据类型。 将词条列表输给
set构造函数， set就会返回一个不重复词表。 首先， 创建一个空集合❶， 然后将每篇文档返回的新词集合添加到该集合中❷。 操作符|用于
求两个集合的并集， 这也是一个按位或（OR） 操作符在数学符号表示上， 按位或操作与集合求并操作使用相同记号
"""
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

""" 3 
获得词汇表后， 便可以使用函数setOfWords2Vec()， 该函数的输入参数为词汇表及某个文档， 输出的是文档向量， 向量的每一元素为1或0，
分别表示词汇表中的单词在输入文档中是否出现。 函数首先创建一个和词汇表等长的向量， 并将其元素都设置为0❸。 接着， 遍历文档中的所有单词，
 如果出现了词汇表中的单词， 则将输出的文档向量中的对应值
设为1。 一切都顺利的话， 就不需要检查某个词是否还在vocabList中， 后边可能会用到这一操作
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

"""
先看看前三个函数的执行效果
"""

def test1():
    listPosts, listClass = loadDataSet()
    mVocabList = createVocabList(listPosts)
    print mVocabList
    setOfWords2Vec(mVocabList, listPosts[0])

# test1()


# ---------------------------训练算法： 从词向量计算概率--------------------------

"""
函数中的输入参数为文档矩阵trainMatrix， 以及由每篇文档类别标签所构成的向量trainCategory。 首先， 计算文档属于侮辱性文档
（class=1） 的概率， 即P(1)。 因为这是一个二类分类问题， 所以可以通过1-P(1)得到P(0)。 对于多于两类的分类问题， 则需要对代码稍加
修改。计算p(wi|c1) 和p(wi|c0)， 需要初始化程序中的分子变量和分母变量❶。 由于w中元素如此众多， 因此可以使用NumPy数组快速计算这些
值。 上述程序中的分母变量是一个元素个数等于词汇表大小的NumPy数组。 在for循环中， 要遍历训练集trainMatrix中的所有文档。 一旦某
个词语（侮辱性或正常词语） 在某一文档中出现， 则该词对应的个数（p1Num或者p0Num） 就加1， 而且在所有的文档中， 该文档的总词数也
相应加1❷。 对于两个类别都要进行同样的计算处理。最后， 对每个元素除以该类别中的总词数❸。 利用NumPy可以很好实现， 
用一个数组除以浮点数即可， 若使用常规的Python列表则难以完成这种任务， 读者可以自己尝试一下。 最后， 函数会返回两个向量和一个概率。
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 1. （以下两行） 初始化概率
    """
    利用贝叶斯分类器对文档进行分类时， 要计算多个概率的乘积以获得文档属于某个类别的概率， 即计算p(w0|1)p(w1|1)p(w2|1)。 如果其中一
    个概率值为0， 那么最后的乘积也为0。 为降低这种影响， 可以将所有词的出现数初始化为1， 并将分母初始化为2
    """
    p0Num = ones(numWords); p1Num = ones(numWords)      # change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        # change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 2. （以下两行） 向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 3. 对每个元素做除法
    """
    另一个遇到的问题是下溢出， 这是由于太多很小的数相乘造成的。 当计算乘积p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时， 由于大部分因子都
    非常小， 所以程序会下溢出或者得到不正确的答案。 （读者可以用Python尝试相乘许多很小的数， 最后四舍五入后会得到0。 ） 一种解决
    办法是对乘积取自然对数。 在代数中有ln(a*b) = ln(a)+ln(b)， 于是通过求对数可以避免下溢出或者浮点数舍入导致的错误。 同时， 采用
    自然对数进行处理不会有任何损失。 图4-4给出函数f(x)与ln(f(x))的曲线。 检查这两条曲线， 就会发现它们在相同区域内同时增加或者减
    少， 并且在相同点上取到极值。 它们的取值虽然不同， 但不影响最终结果。 通过修改return前的两行代码， 将上述做法用到分类器中：
    """
    p1Vect = log(p1Num/p1Denom)          # change to log()
    p0Vect = log(p0Num/p0Denom)          # change to log()
    return p0Vect,p1Vect,pAbusive


def test2():
    listPosts, listClass = loadDataSet()
    # 构建了一个包含所有词的列表mVocabList
    mVocabList = createVocabList(listPosts)
    setOfWords2Vec(mVocabList, listPosts[0])
    trainMat = []
    for postinDoc in listPosts:
        temp = setOfWords2Vec(mVocabList, postinDoc)
        trainMat.append(temp)
    # 文档属于侮辱类的概率pAb
    p0v, p1v, pAb = trainNB0(trainMat, listClass)
    print pAb

"""
接下来看一看在给定文档类别条件下词汇表中单词的出现概率， 看看是否正确。 词汇表中的第一个词是cute， 其在类别0中出现1次， 而在类别1中
从未出现。 对应的条件概率分别为0.041 666 67与0.0。 该计算是正确的。 我们找找所有概率中的最大值， 该值出现在P(1)数组第26个下标位
置， 大小为0.157 894 74。 在myVocabList的第26个下标位置上可以查到该单词是stupid。 这意味着stupid是最能表征类别1（侮辱性文档类）的单词。
"""


"""
代码有4个输入： 要分类的向量vec2Classify以及使用函数trainNB0()计算得到的三个概率。 使用NumPy的数组来计算两个
向量相乘的结果❶。 这里的相乘是指对应元素相乘， 即先将两个向量中的第1个元素相乘， 然后将第2个元素相乘， 以此类推。 接下来将词汇表
中所有词的对应值相加， 然后将该值加到类别的对数概率上。 最后， 比较类别的概率返回大概率对应的类别标签。 这一切不是很难， 对吧？
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 1. 元素相乘 分类计算的核心
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0


"""
对文本做一些修改， 看看分类器会输出什么结果。 这个例子非常简单，但是它展示了朴素贝叶斯分类器的工作原理。
 接下来，我们会对代码做些修改， 使分类器工作得更好。
 函数setOfWords2Vec()稍加修改， 修改后的函数称为bagOfWords2Vec()
 -----------------------------------准备数据： 文档词袋模型---------------------------------------
"""


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # todo 这个词的操作
            returnVec[vocabList.index(word)] += 1
    return returnVec


"""
函数是一个便利函数（convenience function） ， 该函数封装所有操作， 以节省输入
"""


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


# testingNB()

# -----------------------------------使用朴素贝叶斯过滤垃圾邮件----------------------------

"""
准备数据： 切分文本
可以看到， 切分的结果不错， 但是标点符号也被当成了词的一部分。 可以使用正则表示式来切分句子， 其中分隔符是除单词、 数字外的任意字符串
"""
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

""""
函数spamTest()对贝叶斯垃圾邮件分类器进行自动化处理。 导入文件夹spam与ham下的文本文件， 并将它们解析为词列表❶。 接下来
构建一个测试集与一个训练集， 两个集合中的邮件都是随机选出的。 本例中共有50封电子邮件， 并不是很多， 其中的10封电子邮件被随机选择
为测试集。 分类器所需要的概率计算只利用训练集中的文档来完成。Python变量trainingSet是一个整数列表， 其中的值从0到49。 接下
来， 随机选择其中10个文件❷。 选择出的数字所对应的文档被添加到测试集， 同时也将其从训练集中剔除。 这种随机选择数据的一部分作为训
练集， 而剩余部分作为测试集的过程称为留存交叉验证（hold-out crossvalidation） 。 假定现在只完成了一次迭代， 那么为了更精确地估计分类
器的错误率， 就应该进行多次迭代后求出平均错误率。接下来的for循环遍历训练集的所有文档， 对每封邮件基于词汇表并使
用setOfWords2Vec()函数来构建词向量。 这些词在traindNB0()函数中用于计算分类所需的概率。 然后遍历测试集， 对其中每封电子邮件进
行分类❸。 如果邮件分类错误， 则错误数加1， 最后给出总的错误百分比
"""
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)# create vocabulary
    trainingSet = range(50); testSet=[]           # create test set
    # （以下四行） 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        # todo del这个操作步骤
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    # （以下四行） 对测试集分类
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # todo 入参出参的计算方法
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

# ------------------------自动化处理----------------------------------------
spamTest()

#  ----------------------------- 4.7. 示例： 使用朴素贝叶斯分类器从个人广告中获取区域倾向---------------------------

""""
RSS源分类器及高频词去除函数
函数calcMostFreq() ❶。 该函数遍历词汇表中的每个词并统计它在文本中出现的次数， 然后根据出现次数从高到低对词典进行排序，
最后返回排序最高的30个单词。 你很快就会明白这个函数的重要性
以下四行） 计算出现频率
"""
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

""""
函数localWords()使用两个RSS源作为参数。 RSS源要在函数外
导入， 这样做的原因是RSS源会随时间而改变。 如果想通过改变代码来
比较程序执行的差异， 就应该使用相同的输入。 重新加载RSS源就会得
到新的数据， 但很难确定是代码原因还是输入原因导致输出结果的改
变。 函数localWords()与程序清单4-5中的spamTest()函数几乎相
同， 区别在于这里访问的是RSS源❷而不是文件。 然后调用函
数calcMostFreq()来获得排序最高的30个单词并随后将它们移除❸。
函数的剩余部分与spamTest()基本类似， 不同的是最后一行要返回下
面要用到的值。
"""
def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        # 2 每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # （以下四行） 去掉出现次数最高的那些词
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print (item[0])

