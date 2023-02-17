import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import math
import joblib


# Neural Network Design

# Linear Layers
class Linear:
    
    def __init__(self, in_features, out_features):
        np.random.seed(17)
        
        self.weights = np.random.randn(out_features, in_features)
        self.bias= np.random.randn(out_features, 1)
        self.vW = np.zeros((out_features, in_features), dtype=float)
        self.vB = np.zeros((out_features,1), dtype=float)
        self.gW = np.zeros((out_features, in_features), dtype=float)
        self.gB = np.zeros((out_features,1), dtype=float)
        self.iter = 1
        
#     Forward Pass   
    def forward(self, x):
        self.x = x
        self.z = np.dot(self.weights, self.x) + self.bias 
        self.batchsize = self.x.shape[1]
        return self.z
    
#     Backward pass
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma = 0.9, optimiser = "Adam"):
        if optimiser == "SGD":
            out_grad = self.SGD(in_grad = grad, learning_rate = lr, momentum = 0, nesterov = False)
            
        elif optimiser == "Momentum SGD":
            out_grad = self.SGD(in_grad = grad, learning_rate = lr, momentum = momentum, nesterov = False)
            
        elif optimiser == "NAG":
            out_grad = self.SGD(in_grad = grad, learning_rate = lr, momentum = momentum, nesterov = True)
            
        elif optimiser == "Adam":
            out_grad = self.Adam(in_grad = grad, learning_rate = lr, beta = beta , gamma = gamma)
            
        elif optimiser == "AdaGrad":
            out_grad = self.AdaGrad(in_grad = grad, learning_rate = lr) 
            
        elif optimiser == "RMSProp":
            out_grad = self.RMSProp(in_grad = grad, learning_rate = lr, gamma = gamma)
       
        return out_grad

    # SGD Optimiser
    def SGD(self, in_grad, learning_rate = 0.01, momentum = 0, nesterov =  False):
        dw = np.dot(in_grad, self.x.T)
        db = np.mean(in_grad, axis = 1).reshape(-1, 1)
        
        dw = dw/self.batchsize        
        
        self.vW = momentum * self.vW + (1 - momentum) * dw
        self.vB = momentum * self.vB + (1 - momentum) * db 
    
        if nesterov == True:
            tempW = self.weights + momentum * self.vW
            out_grad = np.dot(tempW.T, in_grad)
        else:
            out_grad = np.dot(self.weights.T, in_grad)
        
        self.weights = self.weights -  self.vW * learning_rate
        self.bias = self.bias - self.vB * learning_rate
            
        return out_grad

    # AdaGrad Optimiser
    def AdaGrad(self, in_grad, learning_rate = 0.01):
        eps = 1e-8
        
        dw = np.dot(in_grad, self.x.T)
        db = np.mean(in_grad, axis = 1).reshape(-1, 1)
        
        dw = dw/self.batchsize
        
        self.gB = self.gB + (db**2)
        self.gW = self.gW + (dw**2)
        dw = ( dw / np.sqrt(self.gW + eps)) * learning_rate
        db = ( db / np.sqrt(self.gB + eps)) * learning_rate
        
        self.weights = self.weights - dw
        self.bias = self.bias - db
        
        out_grad = np.dot(self.weights.T, in_grad)
        return out_grad
   
    # RMSProp Optimiser
    def RMSProp(self, in_grad, learning_rate = 0.01, gamma = 0.9):
        eps = 1e-8 
        
        dw = np.dot(in_grad, self.x.T)
        db = np.mean(in_grad, axis = 1).reshape(-1, 1)
        
        dw = dw/self.batchsize
        
        self.gW = gamma * self.gW + (1 - gamma) * (dw**2)
        self.gB = gamma * self.gB + (1 - gamma) * (db**2)
        dw = ( dw / np.sqrt(self.gW + eps)) * learning_rate
        db = ( db / np.sqrt(self.gB + eps)) * learning_rate
        
        self.weights = self.weights - dw
        self.bias = self.bias - db
        
        out_grad = np.dot(self.weights.T, in_grad)
        return out_grad
    
    # Adam Optimiser
    def Adam(self, in_grad, learning_rate = 0.01, beta = 0.9, gamma = 0.9):
        eps=1e-8
        
        dw = np.dot(in_grad, self.x.T)
        db = np.mean(in_grad, axis = 1).reshape(-1, 1)
        
        dw = dw/self.batchsize
        
        self.vW = beta*self.vW + (1 - beta) * dw
        self.vB = beta*self.vB + (1 - beta) * db  
        
        self.gW = gamma * self.gW + (1 - gamma) * (dw**2)
        self.gB = gamma * self.gB + (1 - gamma) * (db**2)
        
        vW1 = self.vW / (1 - (beta ** self.iter))
        vB1 = self.vB / (1 - (beta ** self.iter))
        
        gW1 = self.gW / (1 - (gamma ** self.iter))
        gB1 = self.gB / (1 - (gamma ** self.iter))
        
        self.weights = self.weights - (( vW1 ) / np.sqrt( gW1 + eps )) * learning_rate
        self.bias = self.bias - (( vB1 ) / np.sqrt( gB1 + eps )) * learning_rate
        
        self.iter = self.iter + 1
        out_grad = np.dot(self.weights.T, in_grad)
        return out_grad

# Activation functions
# returns gradient of activation pass and value of activation pass
# sigmoid
class sigmoid:
    def forward(self, x):
        self.x = x
        self.y = expit(self.x)
        return self.y
    
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma=0.9, optimiser = "Adam"):
        return self.y*(1 - self.y)*grad

# ReLu
class ReLu:        
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x >= 0, x, 0)
        return self.y
    
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma=0.9, optimiser = "Adam"):
        return grad * np.where(self.y >= 0, 1, 0)    

# PReLu
class PReLu:
    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x >= 0, self.x, self.alpha*self.x)
        return self.y
    
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma=0.9, optimiser = "Adam"):
        return grad * np.where(self.x >= 0, 1, self.alpha)

# tanh
class tanh:
    def forward(self, x):
        self.x = x
        self.y = np.tanh(self.x)
        return self.y
    
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma=0.9, optimiser = "Adam"):
        return grad * (1 - self.y*self.y)
# softmax
class  softmax:
    def forward(self, x):
        self.x = x - x.max(axis=0, keepdims=True)
        exp = np.exp(self.x)
        self.y = exp / exp.sum(axis=0, keepdims=True)
        return self.y
    
    def backward(self, grad, lr = 0.01, momentum = 0, beta = 0.9, gamma=0.9, optimiser = "Adam"):
        return grad


# Loss
# returns gradient of loss and value of loss
class Loss:
    
    def MSELoss(self, y_true, y_pred):
        self.grad = 2*(y_pred - y_true)
        self.value = np.mean((y_true - y_pred) ** 2)
        return self.value
    
    def CrossEntropyLoss(self, y_true, y_pred):
        eps = 1e-16
        self.grad = y_pred - y_true       
        self.value = np.mean(-1*(y_true * np.log(y_pred + eps)))
        return self.value 
    
    def BCELoss(self, y_true, y_pred):
        eps = 1e-16
        self.value = -1 * np.mean( ( y_true * np.log(y_pred + eps) ) + ((1 - y_true) * (np.log(1 - y_pred + eps))) )
        self.grad = ((1 - y_true) / (1 - y_pred + eps)) - (y_true / (y_pred + eps))              
        return self.value
    
    def backward(self):
        return self.grad

# This Class implements forward propagation, baackward propagation, prediction, saving & loading models
class NeuralNetwork:
    def __init__(self, network):
        self.network = network
        
    # Forward Propagation    
    def forward(self, X):
        for layer in self.network:
            X = layer.forward(X)
        return X
    
    # Back Propagation
    def backprop(self, grad, lr , momentum = 0, beta = 0.9, gamma = 0.9, optimiser = "Adam"):
        for layer in reversed(self.network):
            grad = layer.backward(grad, lr = lr, momentum = momentum, beta = beta, gamma = gamma, optimiser = optimiser)
            
    # Prediction Function
    def predict(self, X, Y):
        y_pred = self.forward(X)
        groundtruth = np.argmax(Y, axis = 0)
        predict = np.argmax(y_pred, axis = 0)
        correct = (groundtruth==predict).sum()
        return (correct*100)/np.size(groundtruth)
    
    # Saving Models
    def save_state_dict(self, filename):
        joblib.dump(self.network, filename)
        
    # Loading Models
    def load_state_dict(self, filename):
        return joblib.load(filename)

    
# Plotting the loss values
def plot(filename, E):
    plt.style.use('ggplot')
    plt.figure(figsize = (8, 5))
    plt.plot(range(len(E)), E)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig(filename+".png")
    plt.show()


# Train function to train the network
def trainNetwork(nn, X_train, Y_train , epoch = 101, lr = 0.001 , momentum = 0, beta = 0.9, gamma = 0.9, optimiser = "SGD",
                 filename= None, save = True):
    E = []
    for e in range(1, epoch):
        for X, Y in zip(X_train, Y_train):
            output = nn.forward(X)
            loss = Loss()
            error = loss.CrossEntropyLoss(Y, output)  
            grad = loss.backward()                 
            nn.backprop(grad = grad, lr =lr, momentum = momentum, beta = beta, gamma = gamma, optimiser = optimiser)
        print(f"Error at Epoch {e} is {error}")
        E.append(error)
        if save:
            nn.save_state_dict(filename+".pkl")
    plot(filename, E)

# Prepares data by doing one hot encoding
def dataloader(X, y):
    no_classes = np.max(y) + 1
    batch_size = y.shape[0]
    Y = np.zeros((batch_size, no_classes), dtype = float)
    for i in range(batch_size):
        Y[i][y[i]] = 1
    X = np.reshape(X, (X.shape[0], 784))
    X = (X - np.min(X))/(np.max(X) - np.min(X))
    X = X.T
    Y = Y.T
    return X, Y

# Dividing Dataset into batches
def batchloader(X, Y, batch_size):
    X_list = []
    Y_list = []
    length = Y.shape[1]
    mod = length % 16
    length -= mod
    start = 0
    end = 0        
    for i in range(0, length, batch_size):
        end = end + batch_size
        X_list.append(X[:, start:end])
        Y_list.append(Y[:, start:end])
        start = start + batch_size
    if mod != 0:
        X_list.append(X[ : , end : end + mod])
        Y_list.append(Y[ : , end : end + mod])
    return X_list, Y_list
        



