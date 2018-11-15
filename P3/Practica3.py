
from scipy.io import loadmat
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np

#Definicion de funciones ------------------------------------------------------------------




#Fin de definicion de funciones -----------------------------------------------------------



#Main

data = loadmat('ex3data1.mat')
y = data['y']  # Representa el valor real de cada ejemplo de entrenamiento de X (y para cada X)
X = data['X']  # Cada fila de X representa una escala de grises de 20x20 desplegada linearmente (400 pixeles)

sample = np.random.choice(X.shape[0], 10)
plt.imshow(X[sample, :].reshape(-1, 20).T)
plt.axis('off')
plt.show()