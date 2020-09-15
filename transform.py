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

