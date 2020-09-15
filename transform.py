import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from multiprocessing import Pool
from time import monotonic
from imageio import imread, imwrite
from skimage import color, filters
from scipy import ndimage

def plotimages(*ims, title=None):
    fig = plt.figure()
    if title:
        plt.title(title)
    for i, im in enumerate(ims):
        ax = fig.add_subplot(1, len(ims), i+1)
        ax.axis('off')
        ax.imshow(im, cmap=cm.gray)
    plt.show()

def map_regions(image):
    return ndimage.label(image)

def do_all_tests():
    test_images = 'test_images/'
    image_paths = [test_images+path for path in os.listdir(test_images)]
    pool = Pool(processes=len(image_paths))
    
    pool.map(do_one, image_paths)
    
    pool.close()

def do_one(path):
    print("Starting ", path)
    im = imread(path)
    gray_im = color.rgb2gray(im)
    
    outlined_im = filters.sobel(gray_im)
    threshed = outlined_im > filters.threshold_otsu(outlined_im)
    threshed = (threshed.astype('uint8'))*255

    final_output = threshed
    plt.imshow(final_output, cmap=cm.gray)
    output = path.split('/')[-1].split('.png')[0]
    output = 'output/'+output+'_output.png'
    imwrite(output, final_output)

    print("Finished: ", path)

if __name__ == "__main__":
    start = monotonic()
    args = sys.argv[1:]
    if len(args) > 0:
        for path in args:
            do_one(path)
    else:
        # do_all_tests()
        do_one('test_images/'+'ricos_grappler.png')
    print("Finished All.")
    print("Elapsed Time ", (monotonic() - start)/60)
