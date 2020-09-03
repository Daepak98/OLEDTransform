from imageio import imread, imwrite
from skimage import feature, filters, color
from matplotlib.pyplot import imshow, axis, colormaps
import os

def outline_edges(path):
    im = imread(path)
    gray_im = color.rgb2gray(im)
    sobel = feature.canny(gray_im, sigma=2.5) * 255
    
    output = path.split('/')[-1].split('.png')[0]
    output = 'output/'+output+'_output.png'
    imwrite(output, sobel.astype('uint8'))
    return im

def draw_bounding_box():
    pass

if __name__ == "__main__":
    test_images = 'test_images/'
    for path in os.listdir(test_images):
        im = outline_edges(test_images+path)