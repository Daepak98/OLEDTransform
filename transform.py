import os
# import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
# from multiprocessing import Pool
# from time import monotonic
from imageio import imread, imwrite
from skimage import color

def plotimages(*ims, title=None):
    fig = plt.figure()
    if title:
        plt.title(title)
    for i, im in enumerate(ims):
        ax = fig.add_subplot(1, len(ims), i+1)
        ax.axis('off')
        ax.imshow(im, cmap=cm.gray)
    plt.show()

oled_folder = "input/oled/"
rgb_folder = "input/rgb/"
choice = "Kat.png"
output_dir = "output/"

oled_temp = color.rgba2rgb(imread(oled_folder+choice))
rgb_temp = color.rgba2rgb(imread(rgb_folder+choice))

plotimages(oled_temp)
plotimages(rgb_temp)