import os
# import sys

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from matplotlib import cm
# from multiprocessing import Pool
# from time import monotonic
from imageio import imread, imwrite
from skimage import color
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from time import monotonic

def preprocess(input_dir, expected_dir):
    input_images = []
    expected_images = []
    files = os.listdir(input_dir)
    limit = 2#len(files)
    for file in files[:limit]:
        temp = color.rgba2rgb(imread(input_dir+file))
        input_images.append(temp)
        temp = color.rgba2rgb(imread(expected_dir+file))
        temp = color.rgb2gray(temp)
        expected_images.append(temp)
    
    # rows = 0
    cols_X = 5 # (loc_0, loc_1, R, G, B)
    cols_Y = 1    
    X = np.zeros((0, cols_X))
    Y = np.zeros((0, cols_Y))
    
    x_i = 0
    for im_i, im in enumerate(input_images):
        exp = expected_images[im_i]
        
        X_temp = im.reshape((im.shape[0]*im.shape[1], 3))
        X_temp = np.hstack((np.zeros((X_temp.shape[0], cols_X - 3)), X_temp))
        X = np.vstack((X, X_temp))
        Y_temp = exp.reshape((exp.shape[0]*exp.shape[1], cols_Y))
        Y = np.vstack((Y, Y_temp))
        
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                X[x_i, :2] = i, j 
                Y[x_i, :] = exp[i, j]
                x_i += 1
        
        return X, Y

def train(x, y):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=42)
    clf = RandomForestRegressor()
    clf = clf.fit(X_train, Y_train.flatten())
    with open('./oledify_model.pkl', 'wb') as out:
        pkl.dump(clf, out)
    return clf, X_test, Y_test

def create_model_using_image(model, im, output):
    new_im = np.zeros_like(im)
    # im_temp = im.reshape((im.shape[0]*im.shape[1], 3))
    inds = np.array(list(np.ndindex(*im.shape[:-1])))
    flat = im.flatten().reshape((-1, 3))
    samples = np.zeros((flat.shape[0], 5))
    samples[:, :2], samples[:, 2:] = inds, flat
    predictions = model.predict(samples)
    new_im = predictions.reshape(im.shape[:-1])
    scaled = (new_im*255).astype('uint8')
    imwrite(output, scaled)
    return scaled
    

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
output_dir = "output/"
model_path = "./oledify_model.pkl"

model = 0
retrain = False
if (not os.path.exists(model_path)) or retrain:
    start = monotonic()
    X, Y = preprocess(rgb_folder, oled_folder)
    print("Preprocessing Time: ", monotonic() - start)
    
    start = monotonic()
    model, x_test, Y_test = train(X, Y)
    print("Training Time: ", monotonic() - start)
else:
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
        
choice = "No Expectations.png"
example = color.rgba2rgb(imread(rgb_folder+choice))
start = monotonic()
create_model_using_image(model, example, output_dir+choice)
print("Writing Image Time: ", monotonic() - start)
