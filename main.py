import os

import matplotlib.pyplot as plt

from time import monotonic
# from skimage import color
import torch
from torch.utils.data import DataLoader
import torchvision
from time import monotonic

from model import OLEDUNet, Train, OLEDDataset


# def create_image_using_model(model, im, output):
#     temp = color.rgba2rgb(im) if im.shape[-1] == 4 else im
#     new_im = np.zeros_like(temp)
#     # im_temp = im.reshape((im.shape[0]*im.shape[1], 3))
#     inds = np.array(list(np.ndindex(*temp.shape[:-1])))
#     flat = temp.flatten().reshape((-1, 3))
#     samples = np.zeros((flat.shape[0], 5))
#     samples[:, :2], samples[:, 2:] = inds, flat
#     predictions = model.predict(samples)
#     new_im = predictions.reshape(im.shape[:-1])
#     scaled = (new_im * 255).astype('uint8')
#     try:
#         imwrite(output, scaled, format=".png")
#     except OSError:
#         print("This doesn't have a good name. Writing to home dir")
#         imwrite("bad_name.png", scaled)
#     return scaled


def plotimages(*ims, title=None):
    fig = plt.figure()
    if title:
        plt.title(title)
    for i, im in enumerate(ims):
        ax = fig.add_subplot(1, len(ims), i + 1)
        ax.axis('off')
        ax.imshow(im, cmap='gray')
    plt.show()


if __name__ == "__main__":
    oled_folder = "input/oled/"
    rgb_folder = "input/rgb/"
    output_dir = "output/"
    model_path = "./oledunet_weights.pth.tar"

    model = OLEDUNet()
    retrain = False
    if (not os.path.exists(model_path)) or retrain:
        print(f"Retraining Model: GPU Available: {torch.cuda.is_available()}")
        ds = OLEDDataset(rgb_folder, oled_folder)
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {
            'num_workers': 4}
        OLEDDataloader = DataLoader(ds, shuffle=True, **kwargs)
        start = monotonic()
        model = Train.Train(OLEDDataloader)
        print("Training Time: ", monotonic() - start)
    with open(model_path, 'rb') as f:
        model_data = torch.load(f)
        model.load_state_dict(model_data['state_dict'])

    # args = sys.argv[1:]
    # ims = []
    # for arg in args:
    #     if (os.path.exists(arg) and os.path.isfile(arg)):
    #         im = imread(arg)
    #         plotimages(im, title=f"Original: {arg}")
    #         ims.append((arg, im))
    #     else:
    #         print(
    #             "Warning! Image at {} does not exist. Will process remaining that do exist.".format(arg))
    # if len(ims) == 0:
    #     choice = "No Expectations.png"
    #     example = color.rgba2rgb(imread(rgb_folder + choice))
    #     ims.append((choice, example))

    # for im in ims:
    #     start = monotonic()
    #     output_name = im[0].split('/')[-1]
    #     output_im = create_image_using_model(
    #         model, im[1], output_dir + output_name)
    #     plotimages(output_im, title="Processed Images")
    #     print("Writing {} Time: ".format(output_name), monotonic() - start)

    model = OLEDUNet()
    test_image = torchvision.io.read_image(os.path.join(rgb_folder, "Joker.png"),
                                           mode=torchvision.io.ImageReadMode.RGB)
    test_image = test_image.type(torch.float32)
    test_image = test_image.reshape((1,) + test_image.shape)
    print(test_image)
    result = model(test_image)[0]
    result = ((result - result.min()) / (result.max() - result.min())) * 255
    result[result < 0.6*255] = 0
    print(result)
    torchvision.io.write_png(result.type(torch.uint8),
                             filename=os.path.join(output_dir, "test.png"))
