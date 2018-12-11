from scipy.io import loadmat
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scopt


def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))


data = loadmat('ex3data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
Y = np.ravel(Y)
X_unos = np.hstack([np.ones((len(X), 1)),X])
nMuestras = len(X)
weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights ['Theta2']


aux = sigmoide(X_unos.dot(theta1.T))
aux = np.hstack([np.ones((len(aux),1)), aux])

#El resultado de utilizar la red neuronal será una matriz de 5000 x 10, con las probabilidades de que cada caso sea un numero.
results = sigmoide(aux.dot(theta2.T))

prediccion = results.argmax(axis = 1)+1 #Este será un array de (1, 5000) con las posibilidades de que un numero haya sido predecido correctamente

Z = (prediccion == Y)*1
probabilidad = sum(Z)/len(Y)

print("La probabilidad de acierto del programa es: ", probabilidad*100, "%")

