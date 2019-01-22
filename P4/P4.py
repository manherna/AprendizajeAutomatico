import displayData
import numpy as np
import displayData as dp
from scipy.io import loadmat
import matplotlib.pyplot as plt

import checkNNGradients as check


# Definicion de funciones -------------------------------------------------------------------------
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoidDerivative(z):
    return sigmoid(z)*(1.0-sigmoid(z))

# Es realmente la función hipótesis. Dada una entrada X, y una red neuronal (thetas), calcula una salida
def forwardProp(thetas1, thetas2, X):
    z2 = thetas1.dot(X.T)
    a2 = sigmoid(z2)
    z3 = thetas2.dot(a2)
    a3 = sigmoid(z3)
    return z2, a2, z3, a3

#CostFun calcula el coste ponderado de una entrada usando front-propagation
def costFun(X, y, theta1, theta2,  reg):
    #Here we assert that we can operate with the parameters
    X = np.array(X)
    y = np.array(y)
    muestras = len(y)

    theta1 = np.array(theta1)
    theta2 = np.array(theta2)

    hipo  = forwardProp(theta1, theta2, X)[3]
    cost = np.sum((-y.T)*(np.log(hipo)) - (1-y.T)*(np.log(1- hipo)))/muestras

    regcost = np.sum(np.power(theta1[1:, 1:], 2)) + np.sum(np.power(theta2[:,1:], 2))
    regcost = regcost * (reg/(2*muestras))

    return cost + regcost

# Este algoritmo calcula el coste de la red neuronal y distribuye el coste entre las neuronas.
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    theta1 = np.insert(theta1,0,1,axis = 0) # Theta1 es un array de num_ocultas +1, num_entradas

    # theta2 es un array de (num_etiquetas, num_ocultas)
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))

    X_unos = np.hstack([np.ones((len(X), 1), dtype = np.int), X])
    nMuestras = len(X)
    y = getYMatrix(y,nMuestras, (max(y)-min(y))+1)

    # Computamos el coste con los thetas obtenidos haciendo uso de la funcion cosfun
    cost = costFun(X_unos, y, theta1, theta2, reg)

    # Forward propagation
    z2, a2, z3, a3 = forwardProp(theta1, theta2, X_unos)



    #Back propagation
    gradW2 = np.zeros(theta2.shape)
    gradW1 = np.zeros(theta1[1:,].shape)

    delta3 = np.array(a3 - y.T)   #(numetiquetas, nmuestra)
    delta2 = (theta2.T.dot(delta3))*sigmoidDerivative(z2)      #(capa2, nmuestras)


    gradW2 = gradW2 + (delta3.dot(a2.T))*(1.0/nMuestras)
    print(delta2[1: , :].shape)
    print(X_unos.shape)
    gradW1 = gradW1 + (delta2[1:, :].dot(X_unos))*(1.0/nMuestras)

    #cosa1 = (1/nMuestras)*gradW1 + (1/nMuestras)*np.append(np.zeros(shape=(theta1.shape[0],1)), theta1[:,1:], axis=1)
    #cosa2 = (1/nMuestras)*gradW2 + (1/nMuestras)*np.append(np.zeros(shape=(theta2.shape[0],1)), theta2[:,1:], axis=1)

    return cost, np.concatenate((gradW1, gradW2), axis = None) # retornamos el coste y los 2 gradientes

# Devuelve la matriz Y de tamaño (nMuestras, nEtiquetas) con filas con todo a 0 menos 1 caso a 1.
def getYMatrix(Y, nMuestras, nEtiquetas):
    nY =  np.zeros((nMuestras, nEtiquetas))
    yaux = np.array(Y) -1
    for i in range(len(nY)):
        nY[i][yaux[i]] = 1
    return nY

#returns an initialized matrix of (1+ L_in, L_out)
def weightInitialize(L_in, L_out):
    cini = 0.12
    aux = np.random.uniform(-cini, cini, size =(L_in, L_out))
    aux = np.insert(aux,0,1,axis = 0)
    return aux


# Fin funciones -----------------------------------------------------------------------------------
data = loadmat('ex4data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
nMuestras = len(X)
Y = np.ravel(Y)


weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights ['Theta2']


params = np.hstack((np.ravel(theta1), np.ravel(theta2)))
print(backprop(params,X.shape[1], 25, 10, X, Y, 1)[0])
#print(check.checkNNGradients(backprop, 1))

#print(check.checkNNGradients(backprop, 1))
