
from scipy.io import loadmat
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as scopt

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

def alg_desGrad_reg(thetas, X, Y, _lambda):
    H = sigmoide(np.dot(X, thetas))  
    return (np.dot((1.0/len(X)), X.T).dot(H-Y))+(_lambda/len(X))*thetas


def fun_coste_regularizada(thetas, X, Y, _lambda):
    H = sigmoide(np.dot(X, thetas))       
    oper1 = -(float(1)/len(X))
    oper2 = np.dot((np.log(H)).T, Y)      
    oper3 = (np.log(1-H)).T                     
    oper4 = 1-Y
    oper5 = (_lambda/(2*len(X)))*np.sum(thetas**2)
    return (oper1 * (oper2 + np.dot(oper3, oper4)))+ oper5

def oneVsAll(X, Y, num_etiquetas, reg):
    thetas = np.zeros([num_etiquetas, X.shape[1]]) #Thetas es un vector de shape (num_etiquetas, 401)

    for i in range(num_etiquetas):
        if(i == 0):
             iaux = 10
        else:
            iaux = i
        
        a = (Y == iaux)*1
        thetas[i] = scopt.fmin_tnc(fun_coste_regularizada, thetas[i], alg_desGrad_reg,args = (X, a, reg))[0]
        
    return thetas

#Fin de definicion de funciones -----------------------------------------------------------



#Main

data = loadmat('ex3data1.mat')
Y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)
nMuestras = len(X)
X_unos = np.hstack([np.ones((len(X), 1)),X])
thetas = np.zeros(len(X_unos[0]))
Y = np.ravel(Y)


thetas_opt = oneVsAll(X_unos, Y, 10, 0.1)
resultados = hipotesis(thetas_opt.T, X_unos) #Resultados es un array de (10, 5000) con el resultado de aplicar hipotesis a X

prediccion = resultados.argmax(axis = 1) #Este será un array de (1, 5000) con las posibilidades de que un numero haya sido predecido correctamente
prediccion[prediccion == 0] = 10

Z = (prediccion == Y)*1
probabilidad = sum(Z)/len(Y)

print("La probabilidad de acierto del programa es: ", probabilidad*100, "%")


#print(Y.shape)
# Para pintar 10 numeros aleatorios de la muestra
#sample = np.random.choice(X.shape[0], 10)
#plt.imshow(X[sample, :].reshape(-1, 20).T)
#plt.axis('off')
#plt.show()