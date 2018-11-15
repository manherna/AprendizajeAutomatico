import random as rnd
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as opt
import csv
import sklearn.preprocessing as skpp


#Definicion de funciones -----------------------------------
def predict_Y (vectX, thetas):
    return np.matmul(vectX, thetas.T)

# Funcion del cuadernillo de practicas 
def fun_coste(thetas, matrizX, vectorY, muestras, _lambda):
    H = sigmoide(np.dot(matrizX, thetas))       
    oper1 = -(float(1)/muestras)
    oper2 = np.dot((np.log(H)).T, vectorY)      
    oper3 = (np.log(1-H)).T                     
    oper4 = 1-vectorY
    oper5 = (_lambda/(2*muestras))*np.sum(thetas**2)

    return (oper1 * (oper2 + np.dot(oper3, oper4)))+ oper5

#diapo 8, descenso de gradiente
def alg_desGrad(thetas, matrizX, vectorY, muestras, _lambda):
    H = sigmoide(np.dot(matrizX, thetas))  
    return (np.dot((1.0/muestras), matrizX.T).dot(H-vectorY))+(_lambda/muestras)*thetas

def sigmoide(z):
    return 1.0/(1.0 + np.exp(-z))


def print_pantalla(X, Y, theta, poly):
    plt.figure()
    pos = np.where(vectorY==1)
    neg = np.where(vectorY==0)

    plt.scatter(X_plain.T[pos, 0], X_plain.T[pos, 1], marker='+', c='k', label = "y = 1")
    plt.scatter(X_plain.T[neg, 0], X_plain.T[neg, 1], marker='o', c='y', label = "y = 0")

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                           np.linspace(x2_min, x2_max))
    h = sigmoide(poly.fit_transform(np.c_[xx1.ravel(),
                                         xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')



    plt.legend()
    #plt.show()
    plt.savefig("Tuvi00001.png")

def evaluate(X, Y, thetas):      
    yEv = predict_Y(X, thetas)
    yEv[ yEv >= 0.5] = 1
    yEv[ yEv < 0.5] = 0
    pcent = np.asarray(np.where(yEv == Y)).size
    pcent = (pcent / len(Y))*100
    print("El porcentaje de aciertos máquina es : ", pcent, "%")


#fin definicion de funciones -----------------------------

# Lectura de datos
file = open('ex2data2.csv',encoding="utf8",errors='ignore')
datos = csv.reader(file)

lineas = np.array(list(datos))
muestras = len(lineas)
vectorY = np.array([0 for n in range(muestras)]) # Este vector se utiliza en la funcion de coste

#Numero de XsubJ que tenemos
numcols = len(lineas[0])
tempa = np.ones(muestras)
X_plain = lineas.T[0:-1].astype(float)            #np.array([[np.zeros(numcols-1)]for y in range (muestras)]))
vectorY = np.hstack(lineas.T[-1]).astype(int)
thetas = np.zeros(28)
_lambda = 0.0001

#Con lambda 1, tenemos un porcentaje de aciertos de 76.27%
#Con lambda 0.5 tenemos un porcentaje de aciertos del 80.5%
#Con lambda 0.25 ""      ""     " "                    81.35%
#Con lambda 0.025 tenemos un porcentaje de aciertos de 83.89%
# 0.0001                                                87.28%

poly = skpp.PolynomialFeatures(6)
matrizX = poly.fit_transform(X_plain.T)
thetas = np.zeros(matrizX.shape[1])

print(matrizX.shape)
print(fun_coste(thetas, matrizX, vectorY, muestras, _lambda))

thetas = np.zeros(matrizX.shape[1])
result = opt.fmin_tnc(func = fun_coste, x0 =thetas, fprime=alg_desGrad, args=(matrizX, vectorY, muestras, _lambda))
thetas_opt = result [0]

print(fun_coste(thetas_opt, matrizX, vectorY, muestras, _lambda))

evaluate(matrizX, vectorY, thetas_opt)
print_pantalla(X_plain.T, vectorY, thetas_opt, poly)
