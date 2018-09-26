# coding=utf-8 #Para que no de fallos de caracteres no-ASCII
# Practica 0 de AAMD
# Implementación en python de la integración por el metodo 
# de Monte Carlo

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# create a new figure
fig = Figure()

# associate fig with the backend
canvas = FigureCanvasAgg(fig)

# add a subplot to the fig
#ax = fig.add_subplot(111)

import numpy as np

x = np.arange(-5,5, 0.1)
y = -x*x #aun falta desplazarlo a la derecha y arriba

plt.figure()
plt.plot(x, y)

# save the figure to test.png
canvas.print_png('grafica.png')

def integra_mc (fun, a, b, num_puntos = 1000):
    # Randomizar una lista con 1000 puntos en el segmento a, b 
    # con la función fun. 
    # Después calcular la integral haciendo la división entre
    # los puntos debajo de la función y fuera
    return 0