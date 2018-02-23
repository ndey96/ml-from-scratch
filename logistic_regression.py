# Logistic Regression

import numpy as np

# x = [# characters, # A]
# y = [french=1, english=0]
x = np.array([[36961, 2503], [43621, 2992], [15694, 1042], [36231, 2487], [29945, 2014], [40588, 2805], [75255, 5062], [37709, 2643], [30899, 2126], [25486, 1784], [37497, 2641], [40398, 2766], [74105, 5047], [76725, 5312], [18317, 1215], [35680, 2217], [42514, 2761], [15162, 990], [35298, 2274], [29800, 1865], [40255, 2606], [74532, 4805], [37464, 2396], [31030, 1993], [24843, 1627], [36172, 2375], [39552, 2560], [72545, 4597], [75352, 4871], [18031, 1119]])
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

def gradient_descent(x, y, mode='batch', alpha=1e-2, max_mistakes=0):
    q = float(len(x))
    w = np.random.randn(3)
    x = normalize_x(x)
    x = np.insert(x, 0, 1, axis=1)
    y = normalize(y)
    if mode == 'batch':
        while num_mistakes(x, y, w) > max_mistakes:
            diff = y - y_hat(x, w)
            for i in range(3):
                w[i] = w[i] + (alpha/q)*np.sum(x[:,i]*diff)
    if mode == 'stochastic':
        while num_mistakes(x, y, w) > max_mistakes:
            diff = y - y_hat(x, w)
            idx = np.random.randint(q)
            for i in range(3):
                w[i] = w[i] + alpha*x[idx,i]*diff[idx]
    print("Final weight vector: {}".format(w))
    print('Number of misclassified examples: {}'.format(num_mistakes(x, y, w)))
    return w

print('x = [# characters, # A]')
print('y = [french=1, english=0]\n')
print('x = {}'.format(x))
print('y = {}\n'.format(y))

print('Performing logistic regression...')
gradient_descent(x, y, mode='batch', alpha=1e-2, max_mistakes=0)
