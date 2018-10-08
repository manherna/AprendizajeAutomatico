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
def funCoste(theta):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras  #Pa vectorizar
    acum = 0.0 #Sumatorio
    for i in range(0, muestras):
        acum = acum + (hipo1v(theta, vectorX[i])-vectorY[i])**2
    
    return acum/2.0*len(vectorX)

def hipo1v(theta, x):
    return theta*x

def hipo2v(theta0, theta1, x):
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
#for i in range (len(tetas1)):
#    costes.append(funCoste(tetas1[i]))
#    print (funCoste(tetas1[i]))

# Hay que usar el algoritmo de descenso de gradiente (pg.27)
# Para theta 0 y 1 hay que actualizar sus valores con este algoritmo. (theta1 en caso de una variable solo)
# Derivada * funcion coste viene "traducido" en la diapositiva 36

#Gradiente de theta0 cuando es = 0 (una variable)
def sum0(theta):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(len(vectorX)):
        acum = acum + (hipo1v(theta, vectorX[i])-vectorY[i])
    
    return acum/len(vectorX) #Sumatorio/muestras

#Gradiente de theta1
def sum1(theta):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(len(vectorX)):
        acum = acum + (hipo1v(theta, vectorX[i])-vectorY[i])*vectorX[i] #Not sure if this is correct (parenthesis)
    
    return acum/len(vectorX) #Sumatorio/muestras

# Funcion para ir actualizando theta0
def alg_desGrad0(theta, alpha):
    return theta - alpha*sum0(theta)

# Funcion para ir actualizando theta1
def alg_desGrad1(theta, alpha):
    return theta - alpha*sum1(theta)

# Con las funciones de descenso de gradiente, ahora solo queda iterar 
# Y actualizar los valores de theta0 y theta1 para que tengan el menor coste posible
# Repeat until convergence 
def fun_final():
    # while
    a = 0.2 # alpha
    temp0 = alg_desGrad0(theta0, a) # theta0 = 0? ir probando en las iteraciones imagino
    temp1 = alg_desGrad1(theta1, a) # theta1 = 1?
    theta0 = temp0
    theta1 = temp1


#print (costes)
####### Intento de grafica 3d
#plt.figure()
#plt.plot(tetas1, costes, '-')
#fig = plt.figure()
#Grafica gradiente 3d
#ax3d = fig.add_subplot(111, projection='3d')
#ax3d.plot_wireframe(tetas0, tetas1, costes) #Aqui da fallo
#plt.show()

plt.figure()
plt.plot(vectorX, vectorY, 'x', color ="red")
plt.xticks(np.arange(min(vectorX),max(vectorX), step=5))
plt.yticks(np.arange(min(vectorY), max(vectorY), step=5))
plt.show()


