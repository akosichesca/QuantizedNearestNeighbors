from LSH import LSH
import numpy as np
from quantize import quantize

def minkowski(x, y, minkowski_p=2, qbits=0):
    #xnorm = np.linalg.norm(x)
    #ynorm = np.linalg.norm(y)
    #x = x/xnorm
    #y = y/ynorm
    #print(x,y)    
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    #print(qbits,x,y)    
    z = np.abs(np.power(x-y,minkowski_p))
    z = quantize(z,qbits)
    z = np.power(np.sum(z),1/minkowski_p)
    return z

def chebyshev(x, y, qbits=0):
    #xnorm = np.linalg.norm(x)
    #ynorm = np.linalg.norm(y)
    #x = x/xnorm
    #y = y/ynorm
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = np.max(x-y)
    return z


def dotproductdist(x, y, qbits=0):
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = quantize(np.multiply(x,y),qbits)
    z = 1-np.sum(z)
    return z

def cosinedist(x, y, qbits=0):
    #xnorm = np.linalg.norm(x)
    #ynorm = np.linalg.norm(y)
    #x = x/xnorm
    #y = y/ynorm
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = quantize(np.multiply(x,y),qbits)
    z = 1-np.sum(z)
    return z

def mcamdist(x,y, qbits=0):
    G = np.zeros(len(x))
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    for i in range(len(x)):        
        if qbits == 4:
            G[i] = conductance.G_4bit[int(y[i])][int(x[i])] 
        if qbits == 3:
            G[i] = conductance.G_3bit[int(y[i])][int(x[i])] 
#0.6*np.exp(conductance.conductance[int(x[i])][int(y[i])]*1e-9/503.236e-12)
    d = np.sum(G)
    return d

def mcam_ideal(x,y, qbits=0):
    G = np.zeros(len(x))
    x = quantize(x,qbits)
    y = quantize(y,qbits)
    z = abs(x - y)  
    for i in range(len(x)):        
        if qbits == 4:
            G[i] = conductance.G_4bit[0][int(z[i])]
        if qbits == 3:
            G[i] = conductance.G_3bit[0][int(z[i])] 
#0.6*np.exp(conductance.conductance[int(x[i])][int(y[i])]*1e-9/503.236e-12)
    d = np.sum(G)
    return d
