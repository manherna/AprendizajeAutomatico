import displayData
import numpy as np
from scipy.io import loadmat



# Definicion de funciones -------------------------------------------------------------------------
def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def hipotesis (thetas, X):
    return sigmoide(X.dot(thetas))

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    theta1 = np.reshape(params_rn[:num_ocultas *(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    theta1 = np.insert(theta1,0,1,axis = 0) # Theta1 es un array de num_ocultas +1, num_entradas
    # theta2 es un array de (num_etiquetas, num_ocultas)
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1): ], (num_etiquetas,(num_ocultas+1)))
    # Array con los thetas de la red.
    # Dimensiones (num_entradas, num_etiquetas)
    red = theta1.T.dot(theta2.T) 
    nMuestras = X.shape[0]
    # Matriz de x de dimensiones (nMuestras (5000 en este caso), num_entradas+1)
    X_unos = np.hstack([np.ones((len(X), 1)),X])

    hipo =  hipotesis(red, X_unos)
    oper1 = -y.T.dot(np.log(hipo))
    oper2 = (1-y).T.dot(1- np.log(hipo))
    print(oper1.shape)
    print(oper2.shape)
    return (1/nMuestras)*(oper1-oper2)
   
    

def getYMatrix(Y, nMuestras, nEtiquetas):
    nY =  np.zeros((nMuestras, nEtiquetas))
    j = 0
    for i in Y:
        if (i == 10): i = 0
        nY[j][i] = 1
        j = j+1
    return nY


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
print(backprop(params,X.shape[1], 25, 10, X, Y_Mat, 0.1))
