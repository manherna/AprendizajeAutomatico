
from scipy.io import loadmat
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

#Definicion de funciones ------------------------------------------------------------------

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def hipotesis (thetas, X):
    return sigmoide(X.dot(thetas))

def funcoste(thetas, X, Y):
    H = sigmoide(np.dot(X, thetas))       
    oper1 = -(float(1)/len(X))
    oper2 = np.dot((np.log(H)).T, Y)      
    oper3 = (np.log(1-H)).T                     
    oper4 = 1-Y
    return oper1 * (oper2 + np.dot(oper3, oper4))

def alg_desGrad(thetas, X, Y):
    H = sigmoide(np.dot(X, thetas))  
    return np.dot((1.0/len(X)), X.T).dot(H-Y)

#Fin de definicion de funciones -----------------------------------------------------------



#Main

data = loadmat('ex3data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
nMuestras = len(X)
X_unos = np.hstack([np.ones((len(X), 1)),X])
thetas = np.zeros(len(X_unos[0]))
Y = np.ravel(Y)


print(funcoste(thetas, X_unos, Y))
#print(Y.shape)




# Para pintar 10 numeros aleatorios de la muestra
#sample = np.random.choice(X.shape[0], 10)
#plt.imshow(X[sample, :].reshape(-1, 20).T)
#plt.axis('off')
#plt.show()