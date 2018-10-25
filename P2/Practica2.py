# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 2 de AAMD
# Regresión Logística
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
#----------- Funciones -----------

def predict_Y (vectX, thetas):
    return np.matmul(vectX, thetas.T)


# Funcion del cuadernillo de practicas 
def fun_coste_vec(matrizX, vectorY, thetas, muestras):
    oper = np.matmul(matrizX, thetas) - vectorY
    return ((np.matmul(oper.T,oper) /(2.0*muestras)))

#diapo 8, descenso de gradiente
def alg_desGrad(thetas, j, rate): #theta[i] y tasa (alpha)
    acum = 0.0 #Sumatorio    
    for i in range(muestras):
        acum = acum + (hipo(X_N[i], thetas.T)-vectorY[i])*X_N[i][j] #Revisar, x es una matriz, no un vector
    
    aux = (float(rate)/muestras)
    return thetas[j] - (aux*acum)

def hipo(x, tetas):
    return x
#--------- Fin funciones ---------

# Lectura de datos
file = open('ex2data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = np.array(list(datos))
muestras = len(lineas)
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Numero de XsubJ que tenemos
numcols = len(lineas[0])
matrizX = np.array([[0 for x in range (numcols-1)]for y in range (muestras)])


for x in range(len(lineas)):
    for y in range (numcols):        
        if(y == numcols-1):
            vectorY[x] = lineas[x][y]            
        else:
            matrizX [x][y] = (float)(lineas[x][y])
            
print("X:", matrizX, "Shape: " ,matrizX.shape)
print("Y:", vectorY, "Shape: " ,vectorY.shape)

#Vector de indices donde Y es positiva
pos = np.where(vectorY==1)

#Grafica
plt.figure()
plt.scatter(matrizX[pos, 0], matrizX[pos, 1], marker='+', c='k')

plt.show()