import displayData
import numpy as np
import displayData as dp
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Definicion de funciones -------------------------------------------------------------------------
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoidDerivative(z):
    return sigmoid(z)*(1-sigmoid(z))

# Es realmente la función hipótesis. Dada una entrada X, y una red neuronal (thetas), calcula una salida
def frontProp(thetas1, thetas2, X):
    z2 = thetas1.dot(X.T)
    a2 = sigmoid(z2)
    z3 = thetas2.dot(a2)
    a3 = sigmoid(z3)
    return z2, a2, z3, a3

#CostFun calcula el coste ponderado de una entrada usando front-propagation
def costFun(X, y, theta1, theta2,  reg):
    hipo  = frontProp(theta1, theta2, X)[3]
    muestras = y.shape[0]
    cost = np.sum((-y.T)*(np.log(hipo)) - (1-y.T)*(np.log(1- hipo)))/muestras
    regcost = ((np.sum(theta1**2)+np.sum(theta2**2))*reg)/(2*muestras)
    return cost + regcost

# Este algoritmo calcula el coste de la red neuronal y distribuye el coste entre las neuronas.
def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    theta1 = np.insert(theta1,0,1,axis = 0) # Theta1 es un array de num_ocultas +1, num_entradas
    # theta2 es un array de (num_etiquetas, num_ocultas)
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))
    # Array con los thetas de la red.
    # Dimensiones (num_entradas, num_etiquetas)
    X_unos = np.hstack([np.ones((len(X), 1), dtype = np.int), X])


    # Computamos el coste con los thetas obtenidos haciendo uso de la funcion cosfun
    cost = costFun(X_unos, y, theta1, theta2, reg)

    deriv = sigmoidDerivative(theta1.dot(X_unos.T))

    z2, a2, z3, a3 = frontProp(theta1, theta2, X_unos)

    gradW2 = np.zeros(theta2.shape)
    gradW1 = np.zeros(theta1[1:,].shape)

    print("W1: ", gradW1.shape)
    #De momento solo está implementada la gradiente de W2, que es Theta2.
    #Aun no está vectorizada. Calcula el coste por cada muestra.

    for i in range (nMuestras):

        delta3 = -(y[i,: ] - a3[:, i])
        delta3 = np.array(delta3)[np.newaxis] # Esto se hace para que numpy lo identifique como un array bidimensional (1, 10)
                                            #   y se pueda trasponer
        casea2 = np.array(a2[:, 1])[np.newaxis] # Lo mismo con éste

        gradW2 = gradW2 + delta3.T.dot(casea2) # Obtenemos el gradiente para este caso y se lo sumamos a lo que llevamos

        delta2 = delta3.dot(theta2)*sigmoidDerivative(z2[:, i])
        xcase = np.array(X_unos[i, :])[np.newaxis]
        gradW1 = gradW1 + delta2[:,1:].T.dot(xcase)


    gradW2 = gradW2/nMuestras #Normalizamos el valor
    gradW1 = gradW1/nMuestras
    return cost, (gradW1, gradW2) # retornamos el coste y los 2 gradientes

# Devuelve la matriz Y de tamaño (nMuestras, nEtiquetas) con filas con todo a 0 menos 1 caso a 1.
def getYMatrix(Y, nMuestras, nEtiquetas):
    nY =  np.zeros((nMuestras, nEtiquetas))
    yaux = Y -1

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
Y_Mat = getYMatrix(Y,nMuestras, (max(Y)-min(Y))+1)


weights = loadmat('ex4weights.mat')
theta1, theta2 = weights['Theta1'], weights ['Theta2']



params = np.hstack((np.ravel(theta1), np.ravel(theta2)))
print(backprop(params,X.shape[1], 25, 10, X, Y_Mat, 1))
