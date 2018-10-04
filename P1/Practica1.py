# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 1 de AAMD
# Regresión lineal
# Manuel Hernandez y Mario Jimenez

import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

import csv

#Función para calcular el tiempo de ejecucion de func
def calctiempo(func, tiempos, nump):
    tic = time.process_time()
    #temp =  func(foo, a, b, int(nump))
    toc = time.process_time()
    tiempos.append(temp)
    return toc - tic

#Calcula la función de coste mediante theta, el numero de muestras
#los vectores de puntos y la función de hipótesis
def funCoste(muestras, vecX, vecY, fHipo, theta):
    #return sum(((fHipo(theta,vecX)-vecY)**2))/2.0*muestras
    acum = 0.0
    for i in range(0, muestras):
        acum = acum + (fHipo(theta, vecX[i])-vecY[i])**2
    
    return acum/2.0*muestras

def funchipo1v(theta, x):
    return theta*x



file = open('ex1data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
    
vectorX = []
vectorY = []
for x in range(len(lineas)):
    vectorX.append(float(lineas[x][0]))
    vectorY.append(float(lineas[x][1]))


tetas = np.arange(-2,5, step = 0.1)
costes = []

# Da para paja
# HAy que vectorizar esta mierda
for i in range (len(tetas)):
    costes.append(funCoste(len(vectorX), vectorX, vectorY, funchipo1v, tetas[i]))
    print (funCoste(len(vectorX), vectorX, vectorY, funchipo1v, tetas[i]))

print (costes)
plt.figure()
plt.plot(tetas, costes, '-')

plt.figure()
plt.plot(vectorX, vectorY, 'x', color ="red")
plt.xticks(np.arange(min(vectorX),max(vectorX), step=5))
plt.yticks(np.arange(min(vectorY), max(vectorY), step=5))
plt.show()

#plt.plot(vectorX, vectorY, 'x') #esto no, igual cambiando la escala, pero demasiados datos, mejor intervalos

#plt.show()