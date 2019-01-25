import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt

# Definicion de funciones -------------------------------------------------------------------------
#Funcion devuelve coste y gradiente de forma regularizada
def hipotesis(thetas, X):
    return X.dot(thetas)

def coste(matrizX, vectorY, thetas, _lambda):
    nMuestras = len(X)   
    hipo = hipotesis(thetas, matrizX).reshape((nMuestras, 1))
    coste = 1.0/(2.0*nMuestras) * ((hipo - vectorY).T).dot((hipo - vectorY)) + (_lambda/(2*len(X)))*thetas[1:].T.dot(thetas[1:])
    return coste

def gradiente(matrizX, vectorY, thetas, _lambda):    
    nMuestras = len(X)
    gradiente = (np.dot((1.0/len(X)), matrizX.T).dot(hipo-vectorY))+(_lambda/len(X))*thetas    
    return gradiente

# Fin funciones -----------------------------------------------------------------------------------
data = loadmat('ex5data1.mat')
y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
#Data set
Xval = data['Xval'] 
yval = data['yval']
#Test set
Xtest = data['Xtest']
ytest = data['ytest']

#Columna de unos
X_unos = np.insert(X, 0, 1, axis=1)
Xval = np.insert(Xval, 0, 1, axis=1)
Xtest = np.insert(Xtest, 0, 1, axis=1)

#Grafica valores X
plt.plot(X, y,'rx')
#plt.show()

Theta = np.array([[1.],[1.]])
print(coste(X_unos, y, Theta, 1))
print(gradiente(X_unos, y, Theta, 1))
Theta_opt = scipy.optimize.fmin_cg(computeCost,x0=myTheta_initial,
                                       fprime=computeGradientFlattened,
                                       args=(myX,myy,mylambda),
                                       disp=print_output,
                                       epsilon=1.49e-12,
                                       maxiter=1000)
#Hallar thetas optimizados

#print(Xval)

