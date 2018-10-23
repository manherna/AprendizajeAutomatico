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


#----------- Funciones -----------

def predict_Y (vectX, thetas):
    return np.matmul(vectX, thetas.T)


# Funcion del cuadernillo de practicas 
def fun_coste_vec(matrizX):
    return np.matmul(((np.transpose(np.matmul(matrizX, thetas) - vectorY))/2*muestras),(np.matmul(matrizX, thetas) - vectorY)) # La multiplicacion de entre medias debe ser '*' o 'np.matmul'?


#diapo 8, descenso de gradiente
def alg_desGrad(thetas, j, rate): #theta[i] y tasa (alpha)
    acum = 0.0 #Sumatorio    
    for i in range(muestras):
        acum = acum + (hipo(X_N[i], thetas.T)-vectorY[i])*X_N[i][j] #Revisar, x es una matriz, no un vector
    
    aux = (float(rate)/muestras)
    return thetas[j] - (aux*acum)

def ec_normal(matrizX, matrizY):
    return np.matmul(np.linalg.pinv(np.matmul(matrizX.T, matrizX)),(np.matmul(matrizX.T, matrizY)))

# Funcion que recibe matriz X con lo ejemplos de entrenamiento
# Para normalizarlos (diapo 11)
def normalizacion(x, media, desv):
    return (x - media)/desv

# Diapo 5, hipotesis
def hipo(x, theta):
    return np.dot(x,np.transpose(theta))



#--------- Fin funciones ---------
# Lectura de datos
file = open('ex1data2.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
muestras = len(lineas)
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Numero de XsubJ que tenemos
numcols = len(lineas[0])
vectorX = np.array([[0 for x in range (numcols)]for y in range (muestras)])

for x in range(len(lineas)):
    for y in range (numcols+1):
        if(y == 0):
            vectorX [x][y] = 1
        elif(y < numcols):
            vectorX [x][y] = lineas[x][y-1]
        else:
            vectorY[x] = lineas[x][y-1]





# Ya que lo tenemos dividido por columnas, calculamos su media (np.mean)
X_T = vectorX.transpose()

mediasX = np.mean(X_T, axis= 1)
mediaY = np.mean(np.array(vectorY))

# Ahora la desviacion estandar (np.std)
desvX = np.std(X_T, axis = 1)
desvY = np.std(vectorY)

# Normalizar datos, sustituyendo cada valor por el cociente entre
# Su diferencia con la media y la desviacion estandar
#COLUMNA 0 de 1os para multiplicar por los theta_0
aux = []
aux.append(X_T[0])
for x in range (1,numcols):
    aux.append(normalizacion(X_T[x], mediasX[x], desvX[x]))

X_T_N = np.array(aux)
X_N = X_T_N.transpose()

print('X_T_N:------------------------------------------------\n',X_T_N)
print('X_N:----------------------------------------------------\n', X_N)

mu = np.array([0.0 for n in range(numcols+1)])
for i in range(numcols):
    mu[i] = mediasX[i]
mu[numcols] = mediaY

sigma = np.array([0.0 for n in range(numcols+1)])
for i in range(numcols):
    sigma[i] = desvX[i]
sigma[numcols] = desvY




thetas = np.array([0.0 for n in range (numcols)])
temps = np.copy(thetas)
alpha = 0.003
for j in range (0, 2000):
    for i in range(len(thetas)):
        temps[i] = alg_desGrad(thetas,i,alpha)
    thetas = np.copy(temps)
    alpha = alpha/3

print(thetas)
xtest = [1, 2200,3]

thetas2 = ec_normal(vectorX, vectorY)
print(predict_Y(xtest, thetas2))
