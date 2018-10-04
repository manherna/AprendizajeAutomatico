# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 1 de AAMD
# Regresión lineal
# Manuel Hernandez y Mario Jimenez

import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

import csv
%precision 2

#Función para calcular el tiempo de ejecucion de func
def calctiempo(func, tiempos, nump):
    tic = time.process_time()
    temp =  func(foo, a, b, int(nump))
    toc = time.process_time()
    tiempos.append(temp)
    return toc - tic

file = open('ex1data1.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = list(datos)
    
vectorX = []
vectorY = []
for x in range(len(lineas)):
    vectorX.append(lineas[x][0])
    vectorY.append(lineas[x][1])

plt.figure()

#plt.plot(vectorX, vectorY, 'x') #esto no, igual cambiando la escala, pero demasiados datos, mejor intervalos

#plt.show()