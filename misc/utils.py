import os, struct
import numpy as np
from array import array as pyarray
from itertools import product as cross
import matplotlib.pyplot as plt



def load_mnist(dataset="training", digits=range(10), path='/Users/aidynkemeldinov/Downloads'):
    
    """
    Load MNIST dataset

    Note: Adapted from: http://cvxopt.org/applications/svm/index.html?highlight=mnist
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = labels.ravel()
    return images, labels


def plot_confusion_matrix(arr, labels, colormap=plt.cm.Reds):
    """
    Adapted from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(arr, cmap=colormap)
    plt.title('Confusion matrix')
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    limit = np.max(arr)/2
    for x, y in cross(range(arr.shape[0]), range(arr.shape[1])):
        plt.text(y, x, format(arr[x, y], 'd'), horizontalalignment="center", color="white" if arr[x, y] > limit else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def binary_plot(X_train,y_train,labelp):

    x_pos = []
    x_neg = []
    for i,v in enumerate(y_train):
        if v==labelp:
            x_pos.append(X_train[i,:])
        else:
            x_neg.append(X_train[i,:])
    x_pos = np.array(x_pos)
    x_neg = np.array(x_neg)

    return x_pos,x_neg

def variance_plot(num_comp,var_exp):

    f, (ax1, ax2) = plt.subplots(2, 1,sharey=True)
    ax1.scatter(num_comp,var_exp,s=0.5,marker="o",facecolor='green')
    ax1.set_title('Total Percentage of Explained Variance vs Number Principal Components')
    ax1.set_ylabel('Explained Variance')
    ax2.scatter(num_comp[:100],var_exp[:100],s=1,marker="o",facecolor='green')
    ax2.axhline(y=90, color='r', linewidth = 0.7, linestyle='-')
    ax2.set_title('Total Percentage of Explained Variance vs Up to First 100 components')
    ax2.set_xlabel('Number of Principle Components')
    ax2.set_ylabel('Explained Variance')
    ax2.legend(['90% explained variance level'])
    ax2.set_xticks(np.arange(0,101,10))
    f.set_size_inches(12, 10)


