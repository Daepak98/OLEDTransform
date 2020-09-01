from imageio import imread, imwrite
from matplotlib.pyplot import imshow, axis, colormaps

# Constants
mid = int(255/2)
black = [0, 0, 0]
white = [255, 255, 255]

def recolor(image):
    for i, row in enumerate(image):
        for j, col in enumerate(row):
            pix = image[i][j]
            # print(pix)
            y = lumino(pix)
            # print(y)
            if y > mid:
                image[i][j] = 255 #white.copy()
            else:
                image[i][j] = 0 #black.copy()

def lumino(pix):
    return .299*pix[0]+.587*pix[1]+.114*pix[2]

if __name__ == "__main__":
    path = 'test_images/saharan_sun.png'
    im = imread(path)
    output = recolor(im)
    imwrite('output.png', im)


# Look at anisotropic filtering