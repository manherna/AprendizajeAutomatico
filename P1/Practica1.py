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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import csv

# Definiciones de Funciones ------------------------------------------------------------

def drawHipothesis(theta0, theta1):
    xxx = np.linspace(min(vectorX), max(vectorX))
    yyy = theta0 + xxx*theta1

    plt.plot(xxx, yyy, '-')


#Calcula la función de coste mediante theta, el numero de muestras
#los vectores de puntos y la función de hipótesis
def funCoste(theta0, theta1):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras  #Pa vectorizar
    acum = 0.0 #Sumatorio
    for i in range(0, muestras):
        acum = acum + ((hipo(theta0, theta1, vectorX[i])-vectorY[i])**2)
    
    return acum/(2.0*muestras)


#Funcion de recta de hipótesis con 2 variables theta0  y theta1
def hipo(theta0, theta1, x):
    return theta0 + theta1*x
    #Gradiente de theta0 cuando es = 0 (una variable)


# Funcion para el descenso de gradiente de theta0
def alg_desGrad0(theta0, theta1, alpha):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(muestras):
        acum = acum + (hipo(theta0, theta1, vectorX[i])-vectorY[i])
    
    temp =  acum/muestras #Sumatorio/muestras
    return theta0 - (alpha*temp)/muestras

# Funcion para el descenso de gradiente de theta1
def alg_desGrad1(theta0, theta1, alpha):
    #return sum(((fHipo(theta,vectorX)-vectorY)**2))/2.0*muestras
    acum = 0.0 #Sumatorio
    for i in range(muestras):
        acum = acum + (hipo(theta0, theta1, vectorX[i])-vectorY[i])*vectorX[i] 
    
    temp = acum/muestras #Sumatorio/muestras
    return theta1 - (alpha*temp)/muestras


def fun_final():
#Haremos 2000 iteraciones para comprobar.
    theta0 = 0.0
    theta1 = 0.0
    x = 2000
    for i in range (0, x):
        a = 0.2
        temp0 = alg_desGrad0(theta0, theta1, a)
        temp1 = alg_desGrad1(theta0, theta1, a)
        theta0 = temp0
        theta1 = temp1

        print('theta0: ',theta0, 'theta1: ', theta1)
        print('Coste: '+ str(funCoste(theta0, theta1)))
        print('\n')

        #Metemos 1 de cada 200 thetas
        if (i == x-1):
            drawHipothesis(theta0, theta1)
    
# Fin de definición de funciones --------------------------------------------------------------




file = open('ex1data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
muestras = len(lineas)
vectorX = []
vectorY = []
for x in range(len(lineas)):
    vectorX.append(float(lineas[x][0]))
    vectorY.append(float(lineas[x][1]))


plt.figure()
plt.plot(vectorX, vectorY, 'x', color ="red")
plt.xticks(np.arange(int(min(vectorX)),int(max(vectorX)), step=5))
plt.yticks(np.arange(int(min(vectorY)), int(max(vectorY)), step=5))
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Ingresos en $10.000")


fun_final()





# Pintado de La función de coste
thetas0 = np.arange(-10, 10, 0.25)
thetas1 = np.arange(-1, 4, 0.25)
thetas0, thetas1 = np.meshgrid(thetas0, thetas1)

costes = funCoste(thetas0, thetas1)

fig = plt.figure()
ax = fig.gca(projection = '3d')

costFun3d = ax.plot_surface(thetas0, thetas1, costes, cmap=cm.coolwarm, linewidth=0, antialiased=False)




# Da para paja (Una variable)
#for i in range (len(tetas1)):
#    costes.append(funCoste(tetas1[i]))
#    print (funCoste(tetas1[i]))

# Hay que usar el algoritmo de descenso de gradiente (pg.27)
# Para theta 0 y 1 hay que actualizar sus valores con este algoritmo. (theta1 en caso de una variable solo)
# Derivada * funcion coste viene "traducido" en la diapositiva 36




#print (costes)
####### Intento de grafica 3d
#plt.figure()
#plt.plot(tetas1, costes, '-')
#fig = plt.figure()
#Grafica gradiente 3d
#ax3d = fig.add_subplot(111, projection='3d')
#ax3d.plot_wireframe(tetas0, tetas1, costes) #Aqui da fallo
#plt.show()


plt.show()


