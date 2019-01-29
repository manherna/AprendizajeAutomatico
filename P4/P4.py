import displayData
import numpy as np
import displayData as dp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt
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
    tuple = (np.ones(len(a2[0])), a2)
    a2 = np.vstack(tuple)
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

    regcost = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:,1:], 2))
    regcost = regcost * (reg/(2*muestras))

    return cost + regcost

# Este algoritmo calcula el coste de la red neuronal y distribuye el coste entre las neuronas.
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, Y, reg):
    th1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    # theta2 es un array de (num_etiquetas, num_ocultas)
    th2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))

    X_unos = np.hstack([np.ones((len(X), 1), dtype = np.float), X])
    nMuestras = len(X)
    y = np.zeros((nMuestras, num_etiquetas))

    y = y + getYMatrix(Y, nMuestras, num_etiquetas)
    # Computamos el coste con los thetas obtenidos haciendo uso de la funcion cosfun
    cost = costFun(X_unos, y, th1, th2, reg)

    # Forward propagation
    z2, a2, z3, a3 = forwardProp(th1, th2, X_unos)

    delta3 = np.array(a3 - y.T)   #(numetiquetas, nmuestra)

    mult = th2.T.dot(delta3)
    mult = np.delete(mult, 0, 0)

    delta2 = (th2.T.dot(delta3))[1: , :] *sigmoidDerivative(z2)  #(capa2, nmuestras)

    gradW1 = np.zeros(th1.shape)
    gradW2 = np.zeros(th2.shape)


    gradW1 = gradW1 + (delta2.dot(X_unos))
    gradW2 = gradW2 + (delta3.dot(a2.T))

    reg1 = (reg/nMuestras)*th1
    reg2 = (reg/nMuestras)*th2


    gradW1 = (gradW1/nMuestras) + reg1
    gradW2 = (gradW2/nMuestras) + reg2

    return cost, np.concatenate((gradW1, gradW2), axis = None) # retornamos el coste y los 2 gradientes



# Devuelve la matriz Y de tamaño (nMuestras, nEtiquetas) con filas con todo a 0 menos 1 caso a 1.
def getYMatrix(Y, nMuestras, nEtiquetas):
    nY =  np.zeros((nMuestras, nEtiquetas))
    yaux = np.array(Y) -1
    for i in range(len(nY)):
        z = yaux[i]
        if(z == 10): z = 0
        nY[i][z] = 1
    return nY

#returns an initialized matrix of (1+ L_in, L_out)
def weightInitialize(L_in, L_out):
    cini = 0.12
    aux = np.random.uniform(-cini, cini, size =(L_in, L_out))
    aux = np.insert(aux,0,1,axis = 0)
    return aux


def NNTest (num_entradas, num_ocultas, num_etiquetas, reg, X, Y, laps):
    t1 = weightInitialize(num_entradas, num_ocultas)
    t2 = weightInitialize(num_ocultas, num_etiquetas)

    params = np.hstack((np.ravel(t1), np.ravel(t2)))
    out = opt.minimize(fun = backprop, x0 = params, args = (num_entradas, num_ocultas, num_etiquetas, X, Y, reg), method='TNC', jac = True, options = {'maxiter': laps})

    Thetas1 = out.x[:(num_ocultas*(num_entradas+1))].reshape(num_ocultas,(num_entradas+1))
    Thetas2 = out.x[(num_ocultas*(num_entradas+1)):].reshape(num_etiquetas,(num_ocultas+1))

    input = np.hstack([np.ones((len(X), 1), dtype = np.float), X])
    hipo = forwardProp(Thetas1, Thetas2, input)[3]


    Ghipo = (hipo.argmax(axis = 0))+1
    print(Ghipo.shape)
    prec = (Ghipo == Y)*1

    precision = sum(prec) / len(X)

    print("Program precision: ", precision *100, "%")




# Fin funciones -----------------------------------------------------------------------------------
data = loadmat('ex4data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
nMuestras = len(X)
Y = np.ravel(Y)


weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights ['Theta2']

NNTest(400, 25, 10, 1, X, Y, 70)
#check.checkNNGradients(backprop, 1)