
from scipy.io import loadmat
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

#Definicion de funciones ------------------------------------------------------------------

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))

def hipotesis (thetas, X):
    print("HIPOTESIS SHAPES: ")
    print("TH: ", thetas.shape)
    print ("X ", X.shape)
    print("SHAPE OF HIPOTHESIS: ",X.dot(thetas.T).shape)


    return X.T.dot(thetas.T)

def funcoste(X, Y, muestras):
    print("FUNCOSTE SHAPES: ")
    print("TH: ", thetas.shape)
    print ("X: ", X.shape)
    print("Y: ", Y.shape)

    oper1 = (1/muestras)
    operlog = np.ravel(np.log(hipotesis(X, thetas)))


    print("operlog ", operlog.shape)
    z = np.ravel(-Y).dot(operlog.T)-np.ravel(1-Y).dot(operlog.T)
    print("Z: ", z.shape)
    return oper1* sum(z)





#Fin de definicion de funciones -----------------------------------------------------------



#Main

data = loadmat('ex3data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
nMuestras = len(X)
thetas = np.zeros(len(X[0]))


print(hipotesis(thetas, X[0]))
print(funcoste(X, Y, nMuestras))
#print(Y.shape)




# Para pintar 10 numeros aleatorios de la muestra
#sample = np.random.choice(X.shape[0], 10)
#plt.imshow(X[sample, :].reshape(-1, 20).T)
#plt.axis('off')
#plt.show()