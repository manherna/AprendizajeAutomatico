# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 0 de AAMD
# Implementación en python de la integración por el metodo 
# de Monte Carlo

import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np


def integra_mc (fun, a, b, num_puntos = 1000):
    # Randomizar una lista con 1000 puntos en el segmento a, b 
    # con la función fun. 
    # Después calcular la integral haciendo la división entre
    # los puntos debajo de la función y fuera
    p_x = []
    func = fun

    p_x = np.arange(a, b, (a-b)%10)

    p_y = [func(n) for n in p_x ]

    maxy = max(p_y)

    #Randomización de puntos de la función
    xx = []
    yy = []
    
    xx = np.array([rnd.randint(0, b) for n in range(0, num_puntos)])
    yy = np.array([rnd.randint(0, maxy) for n in range(0, num_puntos)])

    under = 0
    above = 0.0

    for x in xx:
        if (yy[np.where(xx == x)[0][0]] < func(x)):
            under = under+1

    print (under)

    plt.figure()
    plt.plot(p_x, p_y, '-')
    plt.plot(xx, yy, 'x')
    plt.show()

    return 0





print("Por favor introduzca el intervalo a integrar A B")
inp = input()
a = int(inp.split(' ')[0])
b = int(inp.split(' ')[1])

print(str(a), str(b))

integra_mc(lambda a: a*a, a, b, 1000)

