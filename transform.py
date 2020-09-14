import os
import sys
from multiprocessing import Pool
from time import monotonic

# from matplotlib.pyplot import imshow, axis
import numpy as np
import statistics as st
from imageio import imread, imwrite
from skimage import color, feature, filters


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

def decide_cutoff(row, threshold):
    for i in range(len(row)):
        row[i] = row[i] if row[i] > threshold else 0
    return row

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

    # outlined_im = detect_edges(gray_im, sigma=3.0)
    # outlined_im = detect_edges(gray_im) * 255
    # outlined_im = feature.canny(gray_im) * 255
    outlined_im = filters.sobel(gray_im)
    
    mapped, num_regions = map_regions(outlined_im)

    mapped = mapped

    output = path.split('/')[-1].split('.png')[0]
    output = 'output/'+output+'_output.png'
    imwrite(output, region_mapped.astype('uint8'))

    print("Finished: ", path)

if __name__ == "__main__":
    start = monotonic()
    args = sys.argv[1:]
    if len(args) > 0:
        for path in args:
            do_one(path)
    else:
        # do_all_tests()
        do_one('test_images/'+'journey.png')
    print("Finished All.")
    print("Elapsed Time ", (monotonic() - start)/60)
