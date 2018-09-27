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

    # Sumamos todos los puntos por debajo de la función
    for x in xx:
        if (yy[np.where(xx == x)[0][0]] < fun(x)):
            under = under+1



    plt.figure()
    plt.plot(p_x, p_y, '-')
    plt.plot(xx, yy, 'x')


    return float(under / num_puntos)*(b-a)*maxy

def integra_mc_vector(fun, a, b, num_puntos = 10000):

    under = 0.0


    under = sum (1 for x in xx if fun(x) >= yy[np.where(xx == x)[0][0]])


    plt.figure()
    plt.plot(p_x, p_y, '-')
    plt.plot(xx, yy, 'x')


    return float(under / num_puntos)*(b-a)*maxy


def foo(x):
   # return -np.power(float(x), 2)+100
    return -np.power(float(x-100),2) + 10000

print("Por favor introduzca el intervalo a integrar A B y el numero de puntos")
inp = input()
a = int(inp.split(' ')[0])
b = int(inp.split(' ')[1])
numP = int(inp.split(' ')[2])
fun = foo

# Randomizar una lista con 1000 puntos en el segmento a, b 
# con la función fun. 
# Después calcular la integral haciendo la división entre
# los puntos debajo de la función y fuera
p_x = []


p_x = np.linspace(a,b, 10000)
p_y = [fun(n) for n in p_x ]

maxy = max(p_y)
miny = min(p_y)


#Randomización de puntos de la función
xx = np.array([rnd.uniform(a, float(b)) for n in range(0, numP)])
yy = np.array([rnd.uniform(miny, maxy) for n in range(0, numP)])





tic = 0.0
toc = 0.0

tic = time.process_time()
inte = integra_mc(foo, a, b, numP)
toc = time.process_time()
print ("NORMAL: "+ str(toc-tic) + " INTEGRAL: " + str(inte))

tic = time.process_time()
inte = integra_mc_vector(foo, a, b, numP)
toc = time.process_time()
print ("VECTORIZADA: "+ str(toc-tic)+ " INTEGRAL: " + str(inte))

#print('Montecarlo: '+ str(inte))
print('SciPy' + str(integ.quad(foo, a, b)))

plt.show()





