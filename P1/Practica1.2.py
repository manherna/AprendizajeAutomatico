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
nvars = 2
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Nos creamos el vector X, que almacenará todos los valores para cada XsubN
#
numcols = len(lineas[0])
print(numcols)

vectorX = np.array([[0 for x in range (numcols)]for y in range (muestras)])

for x in range(len(lineas)):
    for y in range (numcols+1):
        if(y == 0):
            vectorX [x][y] = 1
        elif(y < numcols):
            vectorX [x][y] = lineas[x][y-1]
        else:
            vectorY[x] = lineas[x][y-1]

#X = np.array([columna1, columna2]) # Matriz X, después habrá que trasponerla, por
# cómo hemos leido los datos. (tal y como los lee numpy ahora es: 
# arriba la fila 'columna1'(si, el nombre es confuso), y debajo la fila 'columna2')

#medias = np.array(len())



# Ya que lo tenemos dividido por columnas, calculamos su media (np.mean)
X_T = vectorX.transpose()
mediasX = np.mean(X_T, axis= 1)

print ('Medias X: ', mediasX)

mediaY = np.mean(np.array(vectorY))
print('Media Y: ', mediaY)
# Ahora la desviacion estandar (np.std)

desvX = np.std(X_T, axis = 1)
print ('Desviación X: ', desvX)
desvY = np.std(vectorY)
print ('Desviación Y: ', desvY)




print(X_T)
# Normalizar datos, sustituyendo cada valor por el cociente entre
# Su diferencia con la media y la desviacion estandar
#COLUMNA 0 de 1os para multiplicar por los theta_0

for x in range (1,numcols+1):
    X_T[x] = normalizacion(X_T[x], mediasX[x], desvX[x])

print (X_T)
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
