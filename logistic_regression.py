# Logistic Regression

import numpy as np

x = np.array([[36961, 2503], [43621, 2992], [15694, 1042], [36231, 2487], [29945, 2014], [40588, 2805], [75255, 5062], [37709, 2643], [30899, 2126], [25486, 1784], [37497, 2641], [40398, 2766], [74105, 5047], [76725, 5312], [18317, 1215], [35680, 2217], [42514, 2761], [15162, 990], [35298, 2274], [29800, 1865], [40255, 2606], [74532, 4805], [37464, 2396], [31030, 1993], [24843, 1627], [36172, 2375], [39552, 2560], [72545, 4597], [75352, 4871], [18031, 1119]])
# 1's represent french text, 0's represent english text
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def one_hot(a):
    a[a >= 0.5] = 1
    a[a < 0.5] = 0
    return a

def y_hat(x, w):
    logits = sigmoid(np.dot(w,x.T))
    return one_hot(logits)

def num_mistakes(x, y, w):
    return np.sum(np.abs(y_hat(x,w)-y))

def normalize_x(x):
    x0,x1 = normalize(x[:,0]), normalize(x[:,1])
    return np.transpose([x0,x1])

def normalize(a):
    return (a-min(a))/float(max(a)-min(a))

def gradient_descent(x, y, mode='batch', alpha=1e-2, max_mistakes=1):
    q = float(len(x))
    w = np.random.randn(2)
    x_norm = normalize_x(x)
    y_norm = normalize(y)
    if mode == 'batch':
        while num_mistakes(x_norm, y_norm, w) > max_mistakes:
            diff = y_norm - y_hat(x_norm, w)
            w[0] = w[0] + (alpha/q)*np.sum(x_norm[:,0]*diff)
            w[1] = w[1] + (alpha/q)*np.sum(x_norm[:,1]*diff)
    if mode == 'stochastic':
        m = len(x)
        while num_mistakes(x_norm, y_norm, w) > max_mistakes:
            diff = y_norm - y_hat(x_norm, w)
            idx = np.random.randint(m)
            w[0] = w[0] + (alpha/q)*np.sum(x_norm[idx,0]*diff[idx])
            w[1] = w[1] + alpha*x_norm[idx,1]*diff[idx]
    print("Final weight vector: {}".format(w))
    print('Number of misclassified examples: {}'.format(num_mistakes(x_norm, y_norm, w)))
    return w

print('x = [# characters, # A]')
print('y = [french=1, english=0]\n')
print('x = {}'.format(x))
print('y = {}\n'.format(y))

print('Performing logistic regression...')
gradient_descent(x, y, mode='stochastic')
