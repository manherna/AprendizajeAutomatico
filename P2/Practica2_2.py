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


#fin definicicion de funciones -----------------------------

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
thetas = np.zeros(numcols)
_lambda = 1

poly = skpp.PolynomialFeatures(6)
matrizX = poly.fit_transform(X_plain)


print(matrizX.shape)
print(fun_coste(thetas, matrizX, vectorY, muestras, _lambda))

thetas = np.zeros(matrizX.shape[1])
result = opt.fmin_tnc(func = fun_coste, x0 =thetas, fprime=alg_desGrad, args=(matrizX, vectorY, muestras))