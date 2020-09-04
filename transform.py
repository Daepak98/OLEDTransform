from imageio import imread, imwrite
from skimage import color
# from matplotlib.pyplot import imshow, axis
import numpy as np
import statistics as st
import os
from multiprocessing import Pool
from time import monotonic
    
def detect_edges(image, sigma=None):
    edges = np.zeros(image.shape)
    for i, row in enumerate(image):
        highlight_these = highlight_points(row, sigma)
        for val in highlight_these:
            edges[i][val] = 1
    return edges

def highlight_points(vals, sigma=None):
    diffs = []
    indices = []
    length = len(vals)
    for i in range(length-1):
        diff = abs(vals[i+1] - vals[i])
        diffs.append(diff)
    sd = sigma if sigma else st.pstdev(diffs)
    sd_coverage = 4
    m = st.mean(diffs)
    tolerance = [m-sd_coverage*sd, m+sd_coverage*sd] 
    for i, val in enumerate(diffs):
        if val < min(tolerance) or val > max(tolerance):
            indices.append(i)
    return indices

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
    # outlined_im = detect_edges(gray_im, sigma=3.0)
    outlined_im = detect_edges(gray_im) * 255
    
    output = path.split('/')[-1].split('.png')[0]
    output = 'output/'+output+'_output.png'
    imwrite(output, outlined_im.astype('uint8'))

    print("Finished: ", path)

if __name__ == "__main__":
    start = monotonic()
    # do_all_tests()
    do_one('test_images/'+'ricos_grappler.png')
    print("Finished All.")
    print("Elapsed Time ", (monotonic() - start)/60)