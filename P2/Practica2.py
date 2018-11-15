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
import scipy.optimize as opt

import csv
#----------- Funciones -----------

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def pinta_frontera_recta(X, Y, theta):
    #plt.figure()
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
    #plt.savefig("frontera.png")


def predict_Y (vectX, thetas):
    return np.matmul(vectX, thetas.T)

# Funcion del cuadernillo de practicas 
def fun_coste(thetas, matrizX, vectorY, muestras):
    H = sigmoide(np.dot(matrizX, thetas))       
    oper1 = -(float(1)/muestras)
    oper2 = np.dot((np.log(H)).T, vectorY)      
    oper3 = (np.log(1-H)).T                     
    oper4 = 1-vectorY

    return oper1 * (oper2 + np.dot(oper3, oper4))

#diapo 8, descenso de gradiente
def alg_desGrad(thetas, matrizX, vectorY, muestras):
    H = sigmoide(np.dot(matrizX, thetas))  
    return np.dot((1.0/muestras), matrizX.T).dot(H-vectorY)



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
thetas = np.zeros(numcols)



result = opt.fmin_tnc(func = fun_coste, x0 =thetas, fprime=alg_desGrad, args=(matrizX, vectorY, muestras))
thetas_opt = result[0]
print(result[0])

print("Optimized thetas", thetas_opt)
print(" Cost: ", fun_coste(thetas_opt, matrizX, vectorY, muestras))


#Vector de indices donde Y es positiva
plt.figure()
pinta_frontera_recta(X_plain.T, vectorY, thetas_opt)
pos = np.where(vectorY==1)
neg = np.where(vectorY==0)
#Grafica
plt.scatter(X_plain.T[pos, 0], X_plain.T[pos, 1], marker='+', c='k', label = "Admitted")
plt.scatter(X_plain.T[neg, 0], X_plain.T[neg, 1], marker='o', c='y', label = "Not Admited")
plt.legend()

plt.savefig("ScarcedPoints")

yEv = predict_Y(matrizX, thetas_opt)
yEv[ yEv >= 0.5] = 1
yEv[ yEv < 0.5] = 0
print(yEv)

pcent = np.asarray(np.where(yEv == vectorY)).size
pcent = (pcent / len(vectorY))*100
print("El porcentaje de aciertos máquina es : ", pcent, "%")




