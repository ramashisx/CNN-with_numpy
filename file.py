import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import savetxt
from numpy import loadtxt

data = np.array(pd.read_csv('train.csv',sep=','))
data = np.transpose(data)

print('Start')

def softmax(x):
    x = 10*x/np.max(x,axis=1).reshape(-1,1)
    x = np.around(x,decimals=5)
    a = np.exp(x)
    a = np.nan_to_num(a)
    b = np.sum(a,axis=1).reshape(-1,1)
    return a/b

def relu(x):
    x = np.where(x>0,x,0)
    return x

def relu_dev(x):
    x = x > 0
    x = x.astype('uint8')
    return x

class CNN:
    def __init__(self,data,m):
        Images = data[1:]
        Images = np.transpose(Images)
        Images = Images/255
        Digits = data[0:1]
        Digits = np.transpose(Digits)
        fuck = np.zeros(Digits.shape[0]*10).reshape(Digits.shape[0],-1)

        for i in range(Digits.shape[0]):
            fuck[i][Digits[i]] = 1

        X = Images
        Y = fuck
        self.m = m
        self.X_train = X[:m]
        self.Y_train = Y[:m]
        self.X_test = X[m:]
        self.Y_test = Y[m:]

    def init_params(self):
        l = [784,256,128,10]
        W1 = np.random.randn(l[0]*l[1]).reshape(-1,l[1]) 
        B1 = np.random.randn(1*l[1]).reshape(-1,l[1]) 
        W2 = np.random.randn(l[1]*l[2]).reshape(-1,l[2])
        B2 = np.random.randn(1*l[2]).reshape(-1,l[2]) 
        W3 = np.random.randn(l[2]*l[3]).reshape(-1,l[3]) 
        B3 = np.random.randn(1*l[3]).reshape(-1,l[3]) 

        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2
        self.W3 = W3
        self.B3 = B3

    def load_params(self):
        W1 = loadtxt('tW1.csv', delimiter=',')
        B1 = loadtxt('tB1.csv', delimiter=',')

        W2 = loadtxt('tW2.csv', delimiter=',')
        B2 = loadtxt('tB2.csv', delimiter=',')

        W3 = loadtxt('tW3.csv', delimiter=',')
        B3 = loadtxt('tB3.csv', delimiter=',')

        B1 = B1.reshape(1,-1)
        B2 = B2.reshape(1,-1)
        B3 = B3.reshape(1,-1)

        self.W1 = W1
        self.B1 = B1
        self.W2 = W2
        self.B2 = B2
        self.W3 = W3
        self.B3 = B3

    def upload_param(self):

        self.accuracy()
        i = input('Want to upload the above accuracy result ? type "yes" ')
        if i == 'yes':

            savetxt('tW1.csv', self.W1, delimiter=',')
            savetxt('tB1.csv', self.B1, delimiter=',')

            savetxt('tW2.csv', self.W2, delimiter=',')
            savetxt('tB2.csv', self.B2, delimiter=',')

            savetxt('tW3.csv', self.W3, delimiter=',')
            savetxt('tB3.csv', self.B3, delimiter=',')

            print('updated')
        else:
            print('retained')

    def forward_prop(self,X):
        X0 = X
        A0 = X0
        X1 = np.matmul(A0,self.W1)+self.B1
        A1 = relu(X1)
        X2 = np.matmul(A1,self.W2)+self.B2
        A2 = relu(X2)
        X3 = np.matmul(A2,self.W3)+self.B3
        A3 = softmax(X3)

        self.X0 = X0
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.A0 = A0
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3

    def backprop(self):
        dX3 = 2*(self.A3-self.Y_train)
        temp = dX3
        dW3 = 1/self.m*np.matmul(self.A2.T,temp)
        dB3 = 1/self.m*np.sum(temp, axis=0).reshape(1,-1)
        dX2 = np.matmul(self.W3,temp.T)
        temp = dX2.T*relu_dev(self.X2)
        dW2 = 1/self.m*np.matmul(self.A1.T,temp)
        dB2 = 1/self.m*np.sum(temp, axis=0).reshape(1,-1)
        dX1 = np.matmul(self.W2,temp.T)
        temp = dX1.T*relu_dev(self.X1)
        dW1 = 1/self.m*np.matmul(self.A0.T,temp)
        dB1 = 1/self.m*np.sum(temp, axis=0).reshape(1,-1)

        self.dW1 = dW1
        self.dB1 = dB1
        self.dW2 = dW2
        self.dB2 = dB2
        self.dW3 = dW3
        self.dB3 = dB3

    def train(self,n,learning_rate):
        for i in range(n):
            self.forward_prop(self.X_train)
            self.backprop()
            self.update_param(learning_rate)
            if i%10 == 0:
                self.accuracy()
        self.upload_param()

    def update_param(self,learning_rate):
        self.W3 -= learning_rate*self.dW3
        self.B3 -= learning_rate*self.dB3
        self.W2 -= learning_rate*self.W2
        self.B2 -= learning_rate*self.dB2
        self.W1 -= learning_rate*self.dW1
        self.B1 -= learning_rate*self.dB1

    def accuracy(self):
        self.forward_prop(self.X_test)
        A3 = np.argmax(self.A3,axis=1)
        Y0 = np.argmax(self.Y_test,axis=1)
        final = A3-Y0
        print((np.count_nonzero(final == 0)/final.shape[0])*100)
        
test = CNN(data,40000)
test.load_params()
test.train(200,0.01)

print('end')