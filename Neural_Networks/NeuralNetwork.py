import sys
import numpy as np

class Net:
    def __init__(self, lr, nI, nH, nO, epochs, a, m):
        self.lr = lr
        self.nI = nI
        self.nH = nH
        self.nO = nO
        self.epochs = epochs
        self.A = a
        self.m = m
        np.random.seed(24)
        self.W1 = np.random.randn(nI, nH)/10.0
        self.W2 = np.random.randn(nH, nO)/10.0
        self.B  = np.random.randn(nH)
    
    def act1(self, z):
        z[z<0] = 0
        return z

    def act2(self, z):
        N = np.exp(z)
        return N/np.sum(N, axis=1).reshape(-1,1)

    def test(self, X):
        Z1 = np.add(np.matmul(X, self.W1), self.B)
        A1 = self.act1(Z1)
        Z2 = np.matmul(A1, self.W2)
        A2 = self.act2(Z2)
        return A2

    def train(self, X, Y):

        Z1 = np.add(np.matmul(X, self.W1), self.B)
        A1 = self.act1(Z1)
        Z2 = np.matmul(A1, self.W2)
        A2 = self.act2(Z2)
        GT = (Y[:,np.newaxis] == np.arange(10))
        error = A2 - GT

        dW2 = np.matmul(A1.T, error)/X.shape[0] + self.A*self.W2
        dB = np.matmul(error, self.W2.T)*(Z1>0)
        dW1 = np.matmul(X.T, dB)/X.shape[0] + self.A*self.W1

        return dW1, dW2, np.mean(dB, axis=0)

    def loss(self, Y, GT):
        one_hot = (GT[:, np.newaxis] == np.arange(10))
        cross_entropy = -np.mean(np.sum((np.log(Y) * one_hot), axis=1))
        return cross_entropy
    
    def gradient_descent(self, grad, dX):
        tmp = (1-self.m)*np.sum(np.square(dX)) + self.m*grad
        return tmp, dX*(self.lr/np.sqrt(tmp+0.001))
    
    
    def run(self, x_train, y_train):
        
        m1,m2,m3=1,1,1
        for i in range(self.epochs):
        	if i % 101==0:
        		m1,m2,m3=1,1,1
        	dW1, dW2, dB = self.train(x_train, y_train)

        	m1, update = self.gradient_descent(m1, dW1)
        	self.W1 -= update
        	m2, update = self.gradient_descent(m2, dW2)
        	self.W2 -= update
        	m3, update = self.gradient_descent(m3, dB)
        	self.B -= update

        
def get_data(f1, f2, f3):
    TR = np.loadtxt( open(f1, "rb"), delimiter="," )
    TRL= np.loadtxt( open(f2, "rb"))
    TS = np.loadtxt( open(f3, "rb"), delimiter="," )
    
    return TR/255.0, TS/255.0, TRL #, TSL
    
f1, f2, f3 = sys.argv[1], sys.argv[2], sys.argv[3]
x_train, x_test, y_train = get_data(f1, f2, f3)

model = Net(
	lr=0.5,
	nI=784, nH=300, nO=10,
	epochs=300,
	a=1e-6, m=0.99
	)
model.run(x_train, y_train)

predicted = model.test(x_test)
final = np.argmax(predicted, axis=1).astype(np.int)

f = open("test_predictions.csv", "w")
for i in final:
    f.write(str(i)+'\n')
f.close()	