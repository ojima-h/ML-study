import pandas as pd
import math
import itertools
import numpy as np
import pylab as pl
import sys


__author__ = 'hikaru.ojima'


S1 = 'Iris-setosa'
S2 = 'Iris-versicolor'
S3 = 'Iris-virginica'


# D = dimension of data set
# K = class Count
# N = size of data set


def sigmoid(w, x):
    """
    :param w: Weight matrix (D x K)
    :param x: matrix (D x N)
    :return: matrix (K x N)
    """
    e = np.exp(w.T.dot(x))
    return e / e.sum(axis=0)


def differential(y, x, t):
    """
    :param y: matrix (K x N)
    :param x: matrix (D x N)
    :param t: target data (K x N)
    :return: array (KxD dim)
    """
    return np.asarray((y - t).dot(x.T)).flatten()


def hessian_block(x, y, j, k):
    """
    :param x: matrix (D x N)
    :param y: matrix (K x N)
    :return: matrix(D x D)
    """
    i = 1 if j == k else 0
    ya = np.asarray(y)
    return x.dot(np.diag(ya[k, :]).dot(np.diag(i - ya[j, :]))).dot(x.T)


def hessian(y, x):
    """
    :param y: matrix (K x N)
    :param x: matrix (D x N)
    :return: matrix ((KxD) x (KxD))
    """
    d_k, _ = y.shape
    return np.concatenate([
        np.concatenate([hessian_block(x, y, j, k) for j in range(d_k)], axis=0)
        for k in range(d_k)
    ], axis=1)


def regression_once(x, t, w_0, epsilon=0.01):
    """
    :param x: matrix (D x N)
    :param t: matrix (K x N)
    :param w_0: matrix (D x K)
    :return: matrix (D x K)
    """
    d_d, d_k = w_0.shape
    w = w_0
    while True:
        y = sigmoid(w, x)

        f = differential(y, x, t)
        h = hessian(y, x)

        if np.isnan(f).any():
            return

        if np.linalg.norm(f) < epsilon:
            break

        try:
            d = np.linalg.solve(h, f)
            w = w - d.reshape((d_k, d_d)).T
        except np.linalg.linalg.LinAlgError:
            break

    red = np.linalg.norm(differential(sigmoid(w, x), x, t))

    return w.T, red


def regression(x, t, epsilon=0.01, repeat=30):
    """
    :param x: matrix (D x N)
    :param t: matrix (K x N)
    :return: matrix (D x K)
    """
    d_d, _ = x.shape
    d_k, _ = t.shape
    best = None

    for i in range(repeat):
        w_0 = (np.random.rand(d_d, d_k) - 0.5) * .2

        result = regression_once(x, t, w_0, epsilon)
        if result is not None:
            w, r = result
            if best is None or r < best['r']:
                best = {'w': w, 'r': r}

    if best is not None:
        print "residual: %f" % best['r']
        return best['w']
    else:
        print "solution not found"


def normalize(df):
    cov = df.cov()
    l, s = np.linalg.eig(cov)
    return np.diag(l ** -.5).dot(s.T).dot((df - df.mean()).T)



##
# Iris data set
##
def load_data():
    df = pd.read_csv('iris.data', names=['s_length', 's_width', 'p_length', 'p_width', 'species'])
    t = target(df)
    x = points(df)
    return (x, t)


def target(df):
    return np.matrix([
        (df.species == S1).astype('int'),
        (df.species == S2).astype('int'),
        (df.species == S3).astype('int'),
    ])


def points(df):
    return np.array([
        np.repeat(1., df.index.size),  # bias
        df.p_length,
        df.p_width,
    ])


def plot2d3c(x, t, w):
    [w1, w2, w3] = w

    color = [tuple(c) for c in np.asarray(t.T)]
    pl.scatter(x[1], x[2], color=color)

    for (wi, wj) in itertools.combinations([w1,w2,w3], 2):
        [c, a, b] = wi - wj
        px = np.linspace(0, 8)
        py = - (a * px + c) / b
        pl.plot(px, py)

    pl.show()


if __name__ == '__main__':
    x, t = load_data()

    w = regression(x, t)

    plot2d3c(x, t, w)