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

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def zeta(theta, x):
    return np.transpose(theta)*x

def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoide(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.pdf")
    plt.close()

def predict_Y (vectX, thetas):
    return np.matmul(vectX, thetas.T)

# Funcion del cuadernillo de practicas 
def fun_coste(matrizX, vectorY, thetas, muestras):
    H = sigmoide(np.dot(matrizX, thetas))       #(g(Xthetas)
    oper1 = -(1.0/muestras)
    oper2 = np.dot((np.log(H)).T, vectorY)      #((log(g(*thetas)))TRASPUESTA MINIIO
    oper3 = (np.log(1-H)).T                     #(log(1-g(X*thetas))).T
    oper4 = 1-vectorY

    return oper1 * (oper2 + np.dot(oper3, oper4))

#diapo 8, descenso de gradiente
def alg_desGrad(matrizX, vectorY, thetas, m): #theta[i] y tasa (alpha)
    H = sigmoide(np.dot(matrizX, thetas))       #(g(Xthetas)
    oper1 = (1.0/m)*matrizX.T
    oper2 =  H - vectorY

    print(oper1)
    print(oper2)
    print(oper1.dot(oper2))

#--------- Fin funciones ---------

# Lectura de datos
file = open('ex2data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = np.array(list(datos))
muestras = len(lineas)
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Numero de XsubJ que tenemos
numcols = len(lineas[0])
tempa = np.ones(muestras)
X_plain = lineas.T[0:-1].astype(float)
z = (tempa, X_plain)
matrizX = ((np.vstack(z)).T).astype(float)            #np.array([[np.zeros(numcols-1)]for y in range (muestras)]))
vectorY = np.hstack(lineas.T[-1]).astype(int)
tprueba = np.array([0,0,0])

print("Coste: ", fun_coste(matrizX,vectorY, tprueba, muestras))
print(alg_desGrad(matrizX, vectorY, tprueba, muestras))

#print("X:", matrizX, "Shape: " ,matrizX.shape)
#print("Y:", vectorY, "Shape: " ,vectorY.shape)

#Vector de indices donde Y es positiva
pos = np.where(vectorY==1)
neg = np.where(vectorY==0)


#Grafica
plt.figure()
plt.scatter(X_plain.T[pos, 0], X_plain.T[pos, 1], marker='+', c='k')
plt.scatter(X_plain.T[neg, 0], X_plain.T[neg, 1], marker='o', c='y')

plt.show()

