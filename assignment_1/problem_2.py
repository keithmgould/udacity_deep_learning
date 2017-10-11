# Problem 2

# Let's verify that the data still looks good.
# Displaying a sample of the labels and images from the ndarray.
# Hint: you can use matplotlib.pyplot.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from scipy import ndimage
from random import randint
import pdb

#Propriedade dos arquivos salvos.
numImages = 1800
imageSize = 28
pixelDepth = 255.0
pickleImgSet = "/Users/keith/Documents/learn_machine_learning/udacity/deep_learning_nano/assignment_1/notMNIST_small/J.pickle"

def loadImgs(imgSetDir):
    openedData = None
    with open(imgSetDir, "rb") as f:
        openedData = pickle.load(f)

    return (pixelDepth / 2) + openedData * pixelDepth

def printImg(imgBuffer):
    convertedData = (pixelDepth / 2) + imgBuffer * pixelDepth
    plt.imshow(convertedData, cmap='gray')
    plt.show()

def loadSingleImage(imgDir):
    dataset = np.ndarray(shape=(imageSize, imageSize), dtype=np.float32)
    try:
        imageData = (ndimage.imread(imgDir).astype(float) - pixelDepth / 2) / pixelDepth
        if imageData.shape != (imageSize, imageSize):
            raise Exception('Unexpected image shape: %s' % str(imageData.shape))
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    return imageData

imgBuffer = loadImgs(pickleImgSet)

pdb.set_trace()

numRandomImgs = 3
for i in range(numRandomImgs):
    printImg(imgBuffer[randint(0, numImages)])
