import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

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
datos = csv.reader(file)
lineas = np.array(list(datos))
print(len(lineas))

matrizX = lineas.T [0:-1]


displayImage(matrizX.T[0])