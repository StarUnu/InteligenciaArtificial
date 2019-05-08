from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from random import randint, uniform,random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import lagrange

import random
import numpy as np
from scipy.optimize import minimize

global x,y

iris = pd.read_csv("train.csv",sep=",")
#print("iris",iris)
#col_1=iris[[0]]
col_1=np.array(iris[['c1']]).reshape(1,60)[0]
col_2=np.array(iris[['c2']]).reshape(1,60)[0]


x=np.array([col_1,col_2])
y=col_4=np.array(iris[['s']]).reshape(1,60)[0]


##la funcion que se quieren minimizar
def objective(X):
    w2,w1, b = X
    return ((w1)**2+(w2)**2)/2

#el pretexto deveras para minimizarlo
def eq(X):
    w2,w1, b= X
    #y*(b+w*x-1)
    list_w=np.array([w1,w2])
    #p.sum(y*(b+list_w*x-1))
    return b+w2*2.4+w1*4.9 #np.sum(y*(b+np.dot(list_w,x)-1)) #np.sum(list_w)


import autograd.numpy as np
from autograd import grad

def F(L):
    #'Augmented Lagrange function'
    w2,w1,b, _lambda = L
    return objective([w2,w1,b]) - _lambda * eq([w2,w1,b])

# Gradients of the Lagrange function
dfdL = grad(F, 0)

# Find L that returns all zeros in this function.
def obj(L):
    w2,w1,b,_lambda=L
    dFdw2,dFdw1,dFdb,dFdlam=dfdL(L)
    return [dFdw2,dFdw1,dFdb,eq([w2,w1,b])]


class SVM():
	def __init__(self,array_x,array_y,canti_caracteristicas,b1):
		self.array_x=np.array(array_x)
		self.array_y=np.array(array_y)
		lista=[]
		var=0
		while var < canti_caracteristicas:
			lista.append(randint(1,30))
			var+=1

		self.w=np.array(lista)
		self.b=b1
		print("array de thetas",self.w)

	#######esto no se muy bien si esta bien veer videos
	def learn(self,iteraciones,lamnda):
		var=0
		#while var<iteraciones:
		#	var+=1	
		w=lamnda*(self.array_x*self.array_y)
		lista_w=[]
		i=0
		while i < 2:
			lista_w.append(np.sum(w[i]))
			i+=1
		#1/w es el error 
		b=np.sum(lamnda*self.array_y)
		print("esto es el peso de w y b",lista_w,b)
		return lista_w,b

#limite = 100
#array_x=[ e for e in range(limite)]
#array_y=[ e + random() for e  in range(limite)]
array_x=x
array_y=y
a=SVM(array_x,array_y,4,2)
iteraciones=10000


from scipy.optimize import fsolve
w2,w1,b,lambdasvm =fsolve(obj, [0.0,0.0,0.0,1.0] )#

print(f'The answer is at {w2,w1,b,lambdasvm}')

#w,b=a.learn(iteraciones,lambdasvm)

resultado_x=[]
resultado=[]
limite=60
canti=0
resultado=b+w1*array_x[0]+w2*array_x[1]#+w[1]*array_x[1]+w[2]*array_x[2]+w[3]*array_x[3]

'''
while canti<limite:
	resultado_x.append(canti)
	resultado.append(b+w[0]+w[1]*canti)
	canti+=1
'''
w0=np.sum(lambdasvm*(array_x[0]))
w1=np.sum(lambdasvm*(array_x[1]))
b=np.sum(lambdasvm*array_y)

iris = pd.read_csv("test.csv",sep=",")
col_1=np.array(iris[['c1']]).reshape(1,40)[0]
col_2=np.array(iris[['c2']]).reshape(1,40)[0]
#col_3=np.array(iris[['c3']]).reshape(1,40)[0]
#col_4=np.array(iris[['c4']]).reshape(1,40)[0]

#x=np.array([col_1,col_2,col_3,col_4])
x=np.array([col_1,col_2])
y=np.array(iris[['s']]).reshape(1,40)[0]
array_x=x

resultado_test=b+w0*array_x[0]+w1*array_x[1]#w[0]*array_x[0]+w[1]*array_x[1]
array_y=y
i=0
while i< len(array_x[0]):
	if array_y[i]==0:
		plt.scatter(array_x[0][i],array_x[1][i],color='blue')
	else:
		plt.scatter(array_x[0][i],array_x[1][i],color='red')
	plt.scatter(array_x[0][i], resultado_test[i], color='black', linewidth=0.4)
	i=i+1
#plt.plot(array_x[0], resultado_test , color='red', linewidth=1)
#plt.plot(array_x[0], resultado_test+b , color='green', linewidth=1)

#plt.scatter(array_x[int(limite/2):],array_y[int(limite/2):],color='green')
plt.xlabel('Costo de casas ')
plt.ylabel('Area')
plt.show()
