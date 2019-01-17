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

def frontProp(thetas1, thetas2, X):
    z2 = thetas1.dot(X.T)
    a2 = sigmoid(z2)
    z3 = thetas2.dot(a2)
    a3 = sigmoid(z3)
    return a3

def costFun(X, y, theta1, theta2,  reg):
    hipo = frontProp(theta1, theta2, X)
    muestras = y.shape[0]
    cost = np.sum((-y.T)*(np.log(hipo)) - (1-y.T)*(np.log(1- hipo)))/muestras
    regcost = ((np.sum(theta1**2)+np.sum(theta2**2))*reg)/(2*muestras)
    return cost + regcost

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    theta1 = np.insert(theta1,0,1,axis = 0) # Theta1 es un array de num_ocultas +1, num_entradas
    # theta2 es un array de (num_etiquetas, num_ocultas)
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))
    # Array con los thetas de la red.
    # Dimensiones (num_entradas, num_etiquetas)
    X_unos = np.hstack([np.ones((len(X), 1), dtype = np.int), X])
    cost = costFun(X_unos, y, theta1, theta2, reg)

    deriv = sigmoidDerivative(theta1.dot(X_unos.T))
    print(deriv.shape)

    for t in range (len(y)):
        caseX = X_unos[t]
        caseY = y[t]
        hipo = frontProp(theta1, theta2, caseX)
        delta3 = hipo - caseY
        delta2 = theta2.T.dot(delta3).dot(deriv)
        if(t == 1 ): 
            print(delta3.shape)
            print(delta2.shape)
    

    return cost


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
print(backprop(params,X.shape[1], 25, 10, X, Y_Mat, 0.7))
