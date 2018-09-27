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


def integra_mc (fun, a, b, num_puntos = 10000):
    # Randomizar una lista con 1000 puntos en el segmento a, b 
    # con la función fun. 
    # Después calcular la integral haciendo la división entre
    # los puntos debajo de la función y fuera
    p_x = []
    func = fun


    p_x = np.linspace(a,b, 100)
    p_y = [func(n) for n in p_x ]

    maxy = max(p_y)

    #Randomización de puntos de la función
    xx = []
    yy = []

    xx = np.array([rnd.uniform(0.0, float(b)) for n in range(0, num_puntos)])
    yy = np.array([rnd.uniform(0.0, maxy) for n in range(0, num_puntos)])

    under = 0.0

    # Sumamos todos los puntos por debajo de la función
    for x in xx:
        if (yy[np.where(xx == x)[0][0]] < func(x)):
            under = under+1


    plt.figure()
    plt.plot(p_x, p_y, '-')
    plt.plot(xx, yy, 'x')


    return float(under / num_puntos)*(b-a)*maxy



def foo(x):
   # return -np.power(float(x), 2)+100
    return x**2

print("Por favor introduzca el intervalo a integrar A B")
inp = input()
a = int(inp.split(' ')[0])
b = int(inp.split(' ')[1])
inte = integra_mc(foo, a, b, 1000)


print('Montecarlo: '+ str(inte))
print('SciPy' + str(integ.quad(foo, a, b)))

plt.show()





