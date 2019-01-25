import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.optimize as opt

def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(8, 8).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)

def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(8, 8).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)

file = open('Data/optdigits.tra', encoding ="utf8", errors = "ignore")
data = csv.reader(file)
lines = np.array(list(data))

X_input = np.array(lines[:,:-1], dtype = np.int)
Y_input = np.array(lines [:,-1:], dtype = np.int)

numCases = len (X_input)
numTags = len(np.unique(Y_input))

#X matrix with normalized values from [0, 16] to [0.0, 1.0]

X_N = X_input/16.0
Y_real = np.zeros((numCases, numTags), dtype = np.int)
for n in range(numCases):
    Y_real[n][Y_input[n]] = 1
    
displayImage(X_N[0])
print("Caca")