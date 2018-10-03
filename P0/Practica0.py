# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 0 de AAMD
# Implementación en python de la integración por el metodo 
# de Monte Carlo

import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integ
import time



def integra_mc (fun, a, b, num_puntos = 10000):
   
    under = 0.0
    
    xx = np.array([rnd.uniform(a, float(b)) for n in range(0, num_puntos)])
    yy = np.array([rnd.uniform(miny, maxy) for n in range(0, num_puntos)])

    # Sumamos todos los puntos por debajo de la función
    for x in xx:
        if (yy[np.where(xx == x)[0][0]] < fun(x)):
            under = under+1

    return float(under / num_puntos)*(b-a)*maxy

def integra_mc_vector(fun, a, b, num_puntos = 10000):

    under = 0.0
    xx = np.array([rnd.uniform(a, float(b)) for n in range(0, num_puntos)])
    yy = np.array([rnd.uniform(miny, maxy) for n in range(0, num_puntos)])

    under = sum (1 for x in xx if fun(x) >= yy[np.where(xx == x)[0][0]])

    return float(under / num_puntos)*(b-a)*maxy

#Función para calcular el tiempo de ejecucion de func
def calctiempo(func,integ, nump):
    tic = time.process_time()
    integ.append(func(foo, a, b, int(nump)))
    toc = time.process_time()
    return toc - tic

#funcion a integrar
def foo(x):
    return -np.power(float(x), 2)+100
    #return -np.power(float(x-100),2) + 10000



print("Por favor introduzca el intervalo a integrar A B")
inp = input()
a = int(inp.split(' ')[0])
b = int(inp.split(' ')[1])
fun = foo

# Randomizar una lista con 1000 puntos en el segmento a, b 
# con la función fun. 

p_x = []


p_x = np.linspace(a,b, 10000)
p_y = np.array([fun(n) for n in p_x ])

maxy = max(p_y)
miny = min(p_y)



#Arrays para la representación de optimización
valtemps=[]
numpunts=np.linspace(100, 1000, 1000)
inte = []
intevec = []

realinte = integ.quad(foo, a, b)


valtemps = np.array([calctiempo (integra_mc, inte, x) for x in numpunts])
valtempsvec = np.array([calctiempo (integra_mc_vector, intevec, x) for x in numpunts])


print(inte)


#Pintado de resultados
plt.figure()

plt.plot(numpunts, valtemps, '-', label = "Tiempo sin vectorizar", color = "red")
plt.plot(numpunts, valtempsvec, '-', label = "Tiempo con vectorizacion", color = "green")
plt.xlabel("Numero de Puntos")
plt.ylabel("Tiempos de calculo")
plt.legend()

plt.figure()
plt.plot(numpunts, inte, 'x', label = "Integral sin vectorizar")
plt.plot(numpunts,intevec, 'x', label = "Integral vectorizada")
plt.plot(numpunts, [realinte[0] for x in numpunts],'-',  color = "red", label = "Integral Real")
plt.xlabel("Numero de puntos")
plt.ylabel("Valor Integral")
plt.legend()

plt.show()











