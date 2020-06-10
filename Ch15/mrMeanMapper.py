
# coding=utf-8

import sys
from numpy import mat, mean, power

"""
这是一个很简单的例子： 该mapper首先按行读取所有的输入并创建一组对应的浮点数， 然后得到数组的长度并创建NumPy矩阵。 再对所有的值
进行平方， 最后将均值和平方后的均值发送出去。 这些值将用于计算全局的均值和方差。
"""
def read_input(file):
    for line in file:
        yield line.rstrip()
        
input = read_input(sys.stdin)#creates a list of input lines
input = [float(line) for line in input] #overwrite with floats
numInputs = len(input)
input = mat(input)
sqInput = power(input,2)

#output size, mean, mean(square values)
print "%d\t%f\t%f" % (numInputs, mean(input), mean(sqInput)) #calc mean of columns
print >> sys.stderr, "report: still alive" 

# ------------------------下面看看程序的运行效果-------------------------------
# 如果在Windows系统下， 可在DOS窗口输入以下命令: python mrMeanMapper.py < inputFile.txt
# read_input('inputFile.txt')