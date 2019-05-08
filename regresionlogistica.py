from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from random import randint, uniform,random
import random
import sys
import numpy as np
import pandas as pd
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

#saber como leer un archivo *****
############jueves control 
###probar con base de datos iris por que el s me da igual a 0

class RL():
	def __init__(self,canti_caracteristicas,array_x,array_y):
		self.x=array_x
		self.y=array_y
		self.tamano=len(array_x[0])
		lista=[]
		var=0
		
		#np.array([randint(1,30),randint(1,30)])
		#canti_caracteristicas=3;
		while var < canti_caracteristicas:
			lista.append(random.random())
			var+=1
		
		self.Athetas=np.array(lista)
		print("array de thetas",self.Athetas)


	#funcion h
	def h(self,fila):
		#print("esto es otro",np.dot(fila,self.Athetas))
		return np.dot(fila,self.Athetas)

	def s(self,fila):
		print("sigmoidad",1/(1+math.exp(self.h(fila)*-1)))
		return 1/(1+math.exp(self.h(fila)*-1))

	def error(self):
		var=0
		total=Decimal(0.0)
		while var<self.tamano:
			temp_num=0
			if self.s(self.x[var]) !=0:
				temp_num=Decimal(np.log(self.s(self.x[var])) )

			temp_num2=0
			if 1-self.s(self.x[var]) !=0:
				temp_num2=Decimal(np.log(1-self.s(self.x[var])))

			print("temp_num",temp_num)
			print("temp_num2",temp_num2)
			total=total+Decimal((self.y[var]*temp_num)+ (1-self.y[var])*temp_num2 )
			#print("total",self.y[var]*temp_num,(1-self.y[var])*temp_num2)
			var=var+1
		print("total",total)
		return Decimal(-1*1/self.tamano)*total

	#derivada del error deberas serio
	def sumatoria_cambio(self,i):
		var=0
		total=0.0
		while var<self.tamano:
			total=total+(self.s(self.x[var])-self.y[var])*self.x[var][i]
			var=var+1
		return total/self.tamano

	def learn(self,alfa):	
		alfa=0.005#0.000005#0.0004
		tamanoA=len(self.x)
		arreglouno=np.full((1,tamanoA),1) ;
		errorw =self.error()

		umbral=0.010
		print("error",errorw)
		var=0
		anterior=errorw
		while(abs(errorw)>umbral):
			canti_thetas=len(self.Athetas)-1

			while canti_thetas>=0:
				self.Athetas[canti_thetas]=self.Athetas[canti_thetas]-alfa*self.sumatoria_cambio(canti_thetas)
				canti_thetas=canti_thetas-1

			errorw =self.error()
			print("errorw",errorw)
		print("thetas",self.Athetas)
		return self.Athetas
		

iris = pd.read_csv("train.csv",sep=",")
#print("iris",iris)
#col_1=iris[[0]]

col_1=np.array(iris[['c1']]).reshape(1,60)[0]
col_2=np.array(iris[['c2']]).reshape(1,60)[0]
col_3=np.array(iris[['c3']]).reshape(1,60)[0]
col_4=np.array(iris[['c4']]).reshape(1,60)[0]
#,col_3,col_4
array_x=np.array(iris)#np.array([col_1,col_2])
var=0
tamano=len(array_x)
while var<tamano:
	array_x[var][4]=array_x[var][3]
	array_x[var][3]=array_x[var][2]
	array_x[var][2]=array_x[var][1]
	array_x[var][1]=array_x[var][0]
	array_x[var][0]=1
	var+=1

array_y=col_4=np.array(iris[['s']]).reshape(1,60)[0]
regresion=RL(5,array_x,array_y)
Athetas=regresion.learn(0.5)

###########graficarlo
y_predic=[]
x_graficar=[]
i=0
while i<len(array_y):
	if array_y[i]==0:
		plt.scatter(array_x[i][1],array_x[i][2],color='blue')
	else:
		plt.scatter(array_x[i][1],array_x[i][2],color='red')
	y_predic.append(Athetas[0]+array_x[i][1]*Athetas[1]+array_x[i][2]*Athetas[2]+array_x[i][3]*Athetas[3])
	x_graficar.append(array_x[i][1])
	i=i+1
plt.plot(x_graficar,y_predic, color='black', linewidth=0.4)
plt.show()

##########Matriz de confución
array_y=array_y ##etiquetas que deberia de salir
#y_predicnum=y_predic
#y_predic=[]

positivos_falso=0
positivos_positivo=0
falso_falso=0
falso_positivo=0
lista_predic=[]
var=0
while var<len(y_predic):
	if y_predic[var]>=0:##que es positivo es decir clase 1
		lista_predic.append(1)
		if array_y[var]==0:
			positivos_falso+=1
		else:
			positivos_positivo+=1
	else:###que es negativo
		lista_predic.append(0)
		if array_y[var]==0:
			positivos_falso+=1
		else:
			positivos_positivo+=1
	var+=1


###################sensibilidad#########
##sumas columanas
positivas_totales_col=positivos_falso+positivos_positivo
negativos_totales_col=falso_falso+falso_positivo
##lo que hizo veerdadermanete bien nuestro sistema
sensibilidad_positivo_positivo=positivos_positivo/positivas_totales_col
sensibilidad_positivo_negativo=positivos_falso/positivas_totales_col

sensibilidad_negativo_negativo=falso_falso/negativos_totales_col
sensibilidad_negativo_positivo=falso_positivo/negativos_totales_col

##############
#######Exactitud
exactitud=(positivos_positivo+falso_falso)/len(y_predic)
print("Exactitud",exactitud)
print("sensibilidad negativo positivo",sensibilidad_negativo_positivo)
print("sensibilidad negativo positivo",sensibilidad_negativo_positivo)
####Cuanto ha clasificado correctamente las clase de flores de acuerdo 
####a sus caracteristicas 

cm = ConfusionMatrix(array_y, lista_predic)
cm.print_stats()