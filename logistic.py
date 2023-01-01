import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.24)
def sig(z):
    return 1/(1+np.exp(-z))

def loss(y,yh):
    loss=-np.mean(y*(np.log(yh))+((1-y)*np.log(1-yh)))
    return loss
w=0
b=0

yh=sig(w.X+b)

def gradient(x,y,yh):
    m=x.shape[0]
    gw=(1/m)*np.dot(x.T,(yh-y))
    gb=(1/m)*np.sum(yh-y)
    return gw,gb

def normalize(x):
    m,n=x.shape
    for i in range(n):
        x=(x-x.mean(axis=0))/x.std(axis=0)
    return x

def train(X,y,size,nitr,lr):
    m,n=x.shape
    w=np.zeros((n,1))
    b=0
    y=y.reshape((m,1))
    x=normalize(X)
    losses=[]
    for itr in range(nitr):
        for i in range((m-1)//size+1):
            si=i*size
            ei=si+size
            xb=X[si:ei]
            yb=y[si:ei]
            yh=sig(np.dot(xb,w)+b)
            dw,db=gradient(xb,yb,yh)
            w-=lr*dw
            b-=lr*db
        l=loss(y,sig(np.dot(X,w)+b))
        losses.append[l]
    return w,b,losses

def predict(X):
    x=normalize(X)
    pred=sig(np.dot(X,w)+b)
    predc=[]
    predc=[1 if i>0.5 else 0 for i in pred]
    return np.array(predc)

w,b,l = train(X, y, size=100,nitr=1000)
def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy
print(accuracy(X, y_hat=predict(X)))
