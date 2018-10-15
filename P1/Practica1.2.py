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

#--------- Fin funciones ---------
# Lectura de datos
file = open('ex1data2.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
muestras = len(lineas)
columna1 = []
columna2 = []
vectorY = [] # Este vector se utiliza en la funcion de coste
for x in range(len(lineas)):
    columna1.append(float(lineas[x][0]))
    columna2.append(float(lineas[x][1]))
    vectorY.append(float(lineas[x][2]))

X = np.array([columna1, columna2]) # Matriz X, después habrá que trasponerla, por
# cómo hemos leido los datos.

# Ya que lo tenemos dividido por columnas, calculamos su media
mediaX = np.mean(columna1)
mediaY = np.mean(columna2)

# Ahora la desviacion estandar
desvX = np.std(columna1)
desvY = np.std(columna2)

# Funcion que recibe matriz X con lo ejemplos de entrenamiento
# Para normalizarlos (diapo 11)
def normalizacion(x, media, desv):
    return (x - media)/desv

# Normalizar datos, sustituyendo cada valor por el cociente entre
# Su diferencia con la media y la desviacion estandar
# Tres vectores auxiliares que formaran la matriz final
col1 = []
col2 = []

for elem in X[0]: # Primera futura columna (de momento es fila)
    aux = normalizacion(elem, mediaX, desvX)
    col1.append(aux)
for elem in X[1]: # Segunda
    aux = normalizacion(elem, mediaY, desvY)
    col2.append(aux)

X = np.array([col1, col2]) # La matriz final es la traspuesta de esta
X_norm = np.array(X.T) # Matriz normalizada

mu = np.array([mediaX, mediaY])
sigma = np.array([desvX, desvY])

muestras = len(columna1)

# Funcion del cuadernillo de practicas
def fun_coste_vec(matrizX):
    (np.transpose(matrizX - vectorY)*(matrizX - vectorY))/2*muestras

# Introducir diferentes valores para la tasa de aprendizaje
# Y posteriormente construir la grafica de la fun de coste (ir dividiendo entre 3)
tasa = 0.03
#1500 iteraciones?
#for i in range(1500): #FALTA RELLENAR para calcula theta
    # do_stuff()
    # tasa = tasa/3


# Resolver de nuevo el problema con la ecuación normal (sin normalizar atributos (columna1 y 2))
X = np.array([columna1, columna2])
resultado = ec_normal(X) #Esto debería estar ya(?)
def ec_normal(matrizX):
    return np.linalg.pinv(np.matmul(np.transpose(matrizX),matrizX))*np.matmul(np.transpose(matrizX),vectorY)

