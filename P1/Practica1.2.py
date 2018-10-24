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

lineas = np.array(list(datos))
muestras = len(lineas)
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Numero de XsubJ que tenemos
numcols = len(lineas[0])
matrizX = np.array([[0 for x in range (numcols)]for y in range (muestras)])

for x in range(len(lineas)):
    for y in range (numcols+1):
        if(y == 0):
            matrizX [x][y] = 1
        elif(y < numcols):
            matrizX [x][y] = lineas[x][y-1]
        else:
            vectorY[x] = lineas[x][y-1]





# Ya que lo tenemos dividido por columnas, calculamos su media (np.mean)
X_T = matrizX.transpose()

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


thetas = np.array(np.ones(numcols))
temps = np.copy(thetas)
alpha = 1

alphas = []
costes = []



figCostes = plt.figure()
# 10 vueltas en las que dividimos alfa entre 3, para ir probando valores
for i in range (10):
    #1500 iteraciones para el descenso de gradiente
    alphas = []
    costes = []
    for j in range (0, 1500):
        # Actualizacion de thetas temporales
        
        for k in range(len(thetas)):
            temps[k] = alg_desGrad(thetas,k,alpha)
        # copia de los nuevos valores de theta
        thetas = temps
        costes.append(fun_coste_vec(X_N, vectorY, thetas, muestras))
     
    plt.plot(costes, '-', label= "Alpha "+ str(alpha))
    alpha = alpha /3.0

print (thetas)

plt.grid(True)
plt.legend()
plt.show()
#plt.savefig('CostesParaAlphas')

thetas2 = ec_normal(matrizX, vectorY)
xm2 = int(input("Introduzca pies²: "))
xhabs = int(input ("Introduzca el número de habitaciones: "))

xm2_n= normalizacion(xm2,mediasX[1], desvX[1])
xhabs_n = normalizacion(xhabs, mediasX[2], desvX[2])

xtest = [[1, xm2_n, xhabs_n],[1, xm2, xhabs]]

print("Estimación con descenso de gradiente: ", int(predict_Y(xtest[0], thetas)), '$')
print("Estimación con Ecuación: ", int(predict_Y(xtest[1], thetas2)), '$')

