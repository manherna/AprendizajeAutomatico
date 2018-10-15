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

    plt.plot(xxx, yyy, '-', label = 'Recta de hipótesis')
   # linea.set_label("Recta de hipótesis con theta0: "+ str(theta0) + ", theta1: "+str(theta1))

#Calcula la función de coste mediante theta, el numero de muestras
#los vectores de puntos y la función de hipótesis
def funCoste(theta0, theta1, muestras):
    acum = 0.0 #Sumatorio
    for i in range(0, muestras):
        acum = acum + ((hipo(theta0, theta1, vectorX[i])-vectorY[i])**2)
    
    return (acum/(2.0*float(muestras)))

#Funcion de recta de hipótesis con 2 variables theta0  y theta1
def hipo(theta0, theta1, x):
    return theta0 + theta1*x

# Funcion para el descenso de gradiente de theta0
def alg_desGrad0(theta0, theta1, alpha):
    acum = 0.0 #Sumatorio
    for i in range(muestras):
        acum = acum + (hipo(theta0, theta1, vectorX[i])-vectorY[i]) 
    
    return theta0 - ((alpha/muestras)*acum)

# Funcion para el descenso de gradiente de theta1
def alg_desGrad1(theta0, theta1, alpha):
    acum = 0.0 #Sumatorio
    for i in range(muestras):
        acum = acum + (hipo(theta0, theta1, vectorX[i])-vectorY[i])*vectorX[i] 
    
    return theta1 - ((alpha/muestras)*acum)


def fun_final():
#Haremos 1500 iteraciones para comprobar.
    #Inicializacion de theta0 y theta1 a 0.0
    theta0 = 0.0
    theta1 = 0.0
    x = 1500
    a = 0.01
    t0aux = t1aux = 0.0
    costmin = funCoste(theta0, theta1, muestras)
  
    for i in range (0, x):
        temp0 = alg_desGrad0(theta0, theta1, a)
        temp1 = alg_desGrad1(theta0, theta1, a)
        theta0 = temp0
        theta1 = temp1
        costemp = funCoste(theta0, theta1, muestras) 
        if(costmin > costemp):
            t0aux = theta0
            t1aux = theta1
            costmin = costemp

        print('Theta0: ',theta0, 'Theta1: ', theta1)
        print('Coste: ', costemp)
        print('\n')

        #Dibujamos la ultima recta
        if (i == x-1):
            drawHipothesis(theta0, theta1)
        #Retornamos los thetas asociados al valor minimo de la función de coste
    return (t0aux, t1aux)

# Fin de definición de funciones --------------------------------------------------------------



#Lectura de documento
file = open('ex1data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
muestras = len(lineas)
vectorX = []
vectorY = []
for x in range(len(lineas)):
    vectorX.append(float(lineas[x][0]))
    vectorY.append(float(lineas[x][1]))


#Primer pintado de los puntos
plt.figure()
plt.plot(vectorX, vectorY, 'x', color ="red")
plt.xticks(np.arange(int(min(vectorX)),int(max(vectorX)), step=5))
plt.yticks(np.arange(int(min(vectorY)), int(max(vectorY)), step=5))
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Ingresos en $10.000")


#Ejecución de la regresión lineal
taux = fun_final()
plt.legend()
plt.savefig("puntosHipotesis")





# Pintado de La función de coste
thetas0 = np.arange(-10, 10, 0.01)
thetas1 = np.arange(-1, 4, 0.01)
thetas0, thetas1 = np.meshgrid(thetas0, thetas1)

costes = funCoste(thetas0, thetas1, muestras)

fig = plt.figure()
ax = fig.gca(projection = '3d')

costFun3d = ax.plot_surface(thetas0, thetas1, costes, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.xlabel("Valor θ0")
plt.ylabel("Valor θ1")
plt.savefig('3dcoste')
#Función de coste en 2d
cont = plt.figure()
plt.contour(thetas0, thetas1, costes, np.logspace(-2,3,20), cmap=cm.coolwarm)
plt.xlabel("Valor θ0")
plt.ylabel("Valor θ1")
plt.plot(taux[0], taux[1], 'x', color= 'red', label= 'Coste minimo obtenido')
plt.legend()
plt.savefig('Coste2D')

plt.show()


