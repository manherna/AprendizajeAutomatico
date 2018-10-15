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

#TODO: X_0 igual a 1

#----------- Funciones -----------
# Funcion del cuadernillo de practicas (Necesito que la revises Manu)
def fun_coste_vec(matrizX):
    ((np.transpose(np.matmul(matrizX, thetas) - vectorY))/2*muestras)*(np.matmul(matrizX, thetas) - vectorY) # La multiplicacion de entre medias debe ser '*' o 'np.matmul'?


#diapo 8, descenso de gradiente
def alg_desGrad(thetas, j, rate): #theta[i] y tasa (alpha)
    acum = 0.0 #Sumatorio    
    for i in range(muestras):
        acum = acum + (hipo(X_norm[i], thetas)-vectorY[i])*X_norm[i][j] #Revisar, x es una matriz, no un vector
    
    return thetas[j] - ((float(rate)/muestras)*acum)

def ec_normal(matrizX):
    return np.linalg.pinv(np.matmul(np.transpose(matrizX),matrizX))*np.matmul(np.transpose(matrizX),vectorY)

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
columna1 = []
columna2 = []
vectorY = [] # Este vector se utiliza en la funcion de coste
for x in range(len(lineas)):
    columna1.append(float(lineas[x][0]))
    columna2.append(float(lineas[x][1]))
    vectorY.append(float(lineas[x][2]))

#X = np.array([columna1, columna2]) # Matriz X, después habrá que trasponerla, por
# cómo hemos leido los datos. (tal y como los lee numpy ahora es: 
# arriba la fila 'columna1'(si, el nombre es confuso), y debajo la fila 'columna2')

# Ya que lo tenemos dividido por columnas, calculamos su media (np.mean)
mediaX = np.mean(columna1)
mediaY = np.mean(columna2)

# Ahora la desviacion estandar (np.std)
desvX = np.std(columna1)
desvY = np.std(columna2)





# Normalizar datos, sustituyendo cada valor por el cociente entre
# Su diferencia con la media y la desviacion estandar
# Dos vectores auxiliares que formaran la matriz final
col1 = []  #Para guardar normalizacion de los x_1
col2 = []  #Para guardar normalizacion de los x_2

#COLUMNA 0 de 1os para multiplicar por los theta_0
col0 = [1 for n in columna1]
col1 = normalizacion(columna1, mediaX, desvX) #Columna con los valores X_1 normalizados
col2 = normalizacion(columna2, mediaY, desvY) #Columna con los valores X_2 normalizados

X = np.array([col0, col1, col2]) # Matriz [3] [N]. Cada columna
X_norm = np.array(X.T) # Matriz normalizada [N][3]


mu = np.array([mediaX, mediaY])
sigma = np.array([desvX, desvY])

muestras = len(columna1)
#---------HASTA AQUI TODO PRACTICAMENTE BIEN--------


#Ahora viene cuando intentamos averiguar que es theta y me lio

#theta = [] # Hay que tener en cuenta que ahora theta es un vector con tantos elementos n como parametros tiene el caso
# Mi teoria es: tenemos los datos del precio de las casas, que como datos son: tamaño y numero
# De habitaciones, por ultimo el precio (vectorY), entonces... tendriamos dos parametros unicamente?


#Atención a la fumada.
#Cada fila de X_Normalizada, corresponde a un caso de los leídos.
#La función de hipótesis devuelve el valor de y para cada caso de esos
#Aplicando los valores de theta. 

#Cuando hacemos y = theta0+ theta1*x, realmente estamos haciendo 
#                         [1]
# y = [theta0, theta1] *  [x]
# Por tanto, hay que pasar a la función de hipótesis, cada vez, un caso
# concreto del vector de X_T. Y cada uno de esos casos concretos es una
# de las filas del vector. 



# Introducir diferentes valores para la tasa de aprendizaje
# Y posteriormente construir la grafica de la fun de coste (ir dividiendo entre 3)
tasa = 0.03

coste = []
#1500 iteraciones?
#for i in range(1500): #FALTA RELLENAR para calcula theta (vector)
    # coste.append(fun_coste_vec(X[i])) #do_stuff() #vale, como esta funcion llama a las otras para que tasa tenga... importancia?
    # tasa = tasa/3
#------------------------------------------------------------------

#Esto es una formula que viene al final del cuadernillo, se supone que es esa linea y ya
# Resolver de nuevo el problema con la ecuación normal (sin normalizar atributos (columna1 y 2))
X = np.array([columna1, columna2])
#resultado = ec_normal(X) #Esto debería estar ya(?)



thetas = np.array([0 for n in X_norm[0]])
temps = np.copy(thetas)
alpha = 0.0001
for j in range (0, 2000):
    for i in range(len(thetas)):
        temps[i] = alg_desGrad(thetas,i,alpha)
    thetas = np.copy(temps)
    alpha = alpha/3.0

print(thetas)
