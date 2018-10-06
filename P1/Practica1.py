# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 1 de AAMD
# Regresión lineal
# Manuel Hernandez y Mario Jimenez

import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import time

import csv

#Función para calcular el tiempo de ejecucion de func
def calctiempo(func, tiempos, nump):
    tic = time.process_time()
    #temp =  func(foo, a, b, int(nump))
    toc = time.process_time()
    #tiempos.append(temp)
    return toc - tic

#Calcula la función de coste mediante theta, el numero de muestras
#los vectores de puntos y la función de hipótesis
def funCoste(muestras, vecX, vecY, fHipo, theta):
    #return sum(((fHipo(theta,vecX)-vecY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(0, muestras):
        acum = acum + (fHipo(theta, vecX[i])-vecY[i])**2
    
    return acum/2.0*muestras

# Funcion de coste con dos variables
def funCoste2(muestras, vecX, vecY, fHipo, theta0, theta1):
    #return sum(((fHipo(theta,vecX)-vecY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(0, muestras):
        acum = acum + (fHipo(theta0, theta1, vecX[i])-vecY[i])**2
    
    return acum/2.0*muestras

def funchipo1v(theta, x):
    return theta*x

def funchipo2v(theta0, theta1, x):
    return theta0 + theta1*x

file = open('ex1data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
    
vectorX = []
vectorY = []
for x in range(len(lineas)):
    vectorX.append(float(lineas[x][0]))
    vectorY.append(float(lineas[x][1]))

tetas0 = np.arange(-2, 20, step = 0.1)
tetas1 = np.arange(0, 10, step = 0.1) #Pendientes (parece que será positiva)
costes = []

# Da para paja (Una variable)
# Hay que vectorizar esta mierda
#for i in range (len(tetas1)):
#    costes.append(funCoste(len(vectorX), vectorX, vectorY, funchipo1v, tetas1[i]))
#    print (funCoste(len(vectorX), vectorX, vectorY, funchipo1v, tetas1[i]))

#Vamos probando todas las combinaciones
for i in range(len(tetas0)):
    for j in range(len(tetas1)):
        costes.append(funCoste2(len(vectorX), vectorX, vectorY, funchipo2v, tetas0[i], tetas1[j]))

min_coste = min(costes)
print(min_coste)

#print (costes)
plt.figure()
#plt.plot(tetas1, costes, '-')
fig = plt.figure()
#Grafica gradiente 3d
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_wireframe(tetas0, tetas1, costes) #Aqui da fallo
plt.show()

plt.figure()
plt.plot(vectorX, vectorY, 'x', color ="red")
plt.xticks(np.arange(min(vectorX),max(vectorX), step=5))
plt.yticks(np.arange(min(vectorY), max(vectorY), step=5))
plt.show()


