import theano
import theano.tensor as T
import numpy as np
import random

x = T.vector()
w1 = theano.shared(np.array([1.,1.]))
w2 = theano.shared(np.array([1.,1.]))
b1 = theano.shared(1.)
b2 = theano.shared(1.)
z1 = T.dot(w1,x) + b1
z2 = T.dot(w2,x) + b2

a1 = 1/(1 + T.exp(-z1))
a2 = 1/(1 + T.exp(-z2))
w3 = theano.shared(np.array([1.,1.]))
b3 = theano.shared(1.)
z3 = T.dot(w3,[a1,a2]) + b3

y = 1/(1 + T.exp(-z3))
y_hat = T.scalar()
neuron = theano.function(inputs = [x],outputs = y)
cost = T.sum((y-y_hat)**2)
cost_function = theano.function(inputs = [y,y_hat],outputs = cost)

dw1,db1,dw2,db2,dw3,db3 = T.grad(cost,[w1,b1,w2,b2,w3,b3])

gradient = theano.function(
        inputs = [x,y_hat],
        updates = [(w1,w1-1*dw1),(b1,b1-1*db1),
                   (w2,w2-1*dw2),(b2,b2-1*db2),
                   (w3,w3-1*dw3),(b3,b3-1*db3)]
        )
x = [[0,0],[0,1],[1,0],[1,1]]
y_hat = [0,1,1,0]

num = input("請輸入次數:")  #設定為10萬次
for i in range(int(num)):
    print("第",i+1,"次")
    for j in range(4):
        gradient(x[j] , y_hat[j])
for i in range(4):
    print("輸入:",x[i],"結果:",y_hat[i])
    print("neuron輸出:",neuron(x[i]))


print("w1 = ",w1.get_value(),"b1 = ",b1.get_value())
print("w2 = ",w2.get_value(),"b2 = ",b2.get_value())
print("w3 = ",w3.get_value(),"b3 = ",b3.get_value())