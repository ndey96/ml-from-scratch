#Perceptron
import numpy as np

x = np.array([[1, 36961, 2503], [1, 43621, 2992], [1, 15694, 1042], [1, 36231, 2487],
 [1, 29945, 2014], [1, 40588, 2805], [1, 75255, 5062], [1, 37709, 2643],
 [1, 30899, 2126], [1, 25486, 1784], [1, 37497, 2641], [1, 40398, 2766],
 [1, 74105, 5047], [1, 76725, 5312], [1, 18317, 1215], [1, 35680, 2217],
 [1, 42514, 2761], [1, 15162, 990], [1, 35298, 2274], [1, 29800, 1865],
 [1, 40255, 2606], [1, 74532, 4805], [1, 37464, 2396], [1, 31030, 1993],
 [1, 24843, 1627], [1, 36172, 2375], [1, 39552, 2560], [1, 72545, 4597],
 [1, 75352, 4871], [1, 18031, 1119]])
# 1's represent french text, 0's represent english text
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def threshold(w, x):
    z = np.dot(w,x)
    if z >= 0:
        return 1
    else:
        return 0

def num_mistakes(x, y, w):
    num = 0
    for i in range(len(x)):
        num = num + np.abs(threshold(w,x[i])-y[i])
    return num

def learning_rule(x, y, max_mistakes=1):
    w = np.random.randn(3)
    m = len(x)
    while num_mistakes(x, y, w) > max_mistakes:
        i = np.random.randint(m)
        if y[i] != threshold(w,x[i]):
            w[0] = w[0] + (y[i]-threshold(w,x[i]))*x[i][0]
            w[1] = w[1] + (y[i]-threshold(w,x[i]))*x[i][1]
            w[2] = w[2] + (y[i]-threshold(w,x[i]))*x[i][2]
    print("Final w: {}".format(w))
    print('Number of misclassified examples: {}'.format(num_mistakes(x, y, w)))
    return w

learning_rule(x, y)
