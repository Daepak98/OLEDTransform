from imageio import imread, imwrite
from matplotlib.pyplot import imshow, axis, colormaps
import statistics as st

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
                image[i][j] = 255 #hite.copy()
            else:
                image[i][j] = 0 #black.copy()

def lumino(pix):
    return .299*pix[0]+.587*pix[1]+.114*pix[2]

def detect_edges(image):
    highlight_color = [3, 254, 129]
    rows = len(image)
    for i, row in enumerate(image[:rows]):
        lumosity = []
        for col in row:
            lumosity.append(lumino(col))
        highlight_these = highlight_points(lumosity)
        for val in highlight_these:
            image[i][val] = highlight_color.copy()
        print(i)
    print('done detect_edges')


def highlight_points(vals):
    diffs = []
    indices = []
    for i in range(len(vals)-1):
        diff = abs(vals[i+1] - vals[i])
        diffs.append(diff)
        i += 2
    m, sd = st.mean(diffs), st.pstdev(diffs)
    tolerance = [m-2*sd, m+2*sd] 
    for i, val in enumerate(diffs):
        if val < min(tolerance) or val > max(tolerance):
            indices.append(i)
    return indices

if __name__ == "__main__":
    path = 'test_images/saharan_sun.png'
    im = imread(path)
    # output = recolor(im)
    detect_edges(im)
    imwrite('output.png', im)


# Look at anisotropic filtering