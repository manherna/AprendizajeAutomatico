import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt

# Definicion de funciones -------------------------------------------------------------------------
#Funcion hipotesis lineal
def hipotesis(X, thetas):
    return X.dot(thetas)

def coste(thetas, matrizX, vectorY, _lambda=0.): 
    nMuestras = len(X)
    hipo = hipotesis(matrizX, thetas).reshape((nMuestras,1))
    cost = (1.0/(2*nMuestras) * ((hipo-vectorY).T.dot(hipo-vectorY))) + (float(_lambda)/(2*nMuestras)) * float(thetas[1:].T.dot(thetas[1:]))
    return cost

def gradiente(thetas, matrizX, vectorY, _lambda=0.):
    thetas = thetas.reshape((thetas.shape[0],1))
    nMuestras = len(X)    
    grad = (1.0/nMuestras)*matrizX.T.dot(hipotesis(matrizX, thetas)-vectorY) + _lambda/(nMuestras)*thetas
    return grad 
    
def gradiente_min(thetas, matrizX, vectorY, _lambda=0.):
    return gradiente(thetas, matrizX, vectorY, _lambda=0.).flatten()
    
# Fin funciones -----------------------------------------------------------------------------------
data = loadmat('ex5data1.mat')
y = data['y']
X = data['X'] 
#Data set
Xval = data['Xval'] 
yval = data['yval']
#Test set
Xtest = data['Xtest']
ytest = data['ytest']

#Columna de unos
X_unos = np.insert(X, 0, 1, axis=1)

plt.plot(X, y,'rx')
#plt.show

Theta = np.ones((2, 1))
print(coste(Theta, X_unos, y, 1.0))
print(gradiente(Theta, X_unos, y, 1.0))

lamb = 0.0
Theta_opt = opt.fmin_cg(coste, Theta, gradiente_min,(X_unos, y, lamb),True, 1.49e-12, 200)

