import numpy as np
from quantize import quantize

def minkowski(x, y):
    xnorm = np.linalg.norm(x)
    ynorm = np.linalg.norm(y)
    x = x/xnorm
    y = y/ynorm
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = np.abs(np.power(x-y,minkowski_p))
    z = np.power(z,1/minkowski_p)
    z = quantize(z,qbits)
    z = np.sum(z)
    return z


def chebyshev(x, y):
    xnorm = np.linalg.norm(x)
    ynorm = np.linalg.norm(y)
    x = x/xnorm
    y = y/ynorm
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = np.max(np.abs(x-y))
    return z


def dotproductdist(x, y):
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = np.multiply(x,y)
    if (qbits == 0):
        z = 1-np.sum(z)
    else:
        z = 1/(np.sum(z)+0.01)
    return z

def cosinedist(x, y):
    xnorm = np.linalg.norm(x)
    ynorm = np.linalg.norm(y)
    x = x/xnorm
    y = y/ynorm
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = np.multiply(x,y)
    if (qbits == 0):
        z = 1-np.sum(z)
    else:
        z = 1/(np.sum(z)+0.01)
    return z

def mcamdist(x,y):
    G = np.zeros(len(x))
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    for i in range(len(x)):
        if qbits == 4:
            G[i] = conductance.G_4bit[int(y[i])][int(x[i])]
        if qbits == 3:
            G[i] = conductance.G_3bit[int(y[i])][int(x[i])]
    d = np.sum(G)
    return d

def mcam_ideal(x,y):
    G = np.zeros(len(x))
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = abs(x - y)
    for i in range(len(x)):
        if qbits == 4:
            G[i] = conductance.G_4bit[0][int(z[i])]
        if qbits == 3:
            G[i] = conductance.G_3bit[0][int(z[i])]
    d = np.sum(G)
    return d
