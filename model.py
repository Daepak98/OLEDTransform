import os

# import matplotlib.pyplot as plt

# from matplotlib import cm
# from multiprocessing import Pool
# from time import monotonic
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision


class OLEDUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, bn_momentum=0.5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn_momentum = bn_momentum

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv2d(
            self.in_channels, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=bn_momentum)

        # self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.BNEn31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        # self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.BNEn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        # self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.BNEn33 = nn.BatchNorm2d(256, momentum=bn_momentum)

        # self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.BNEn41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNEn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNEn43 = nn.BatchNorm2d(512, momentum=bn_momentum)

        # self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNEn51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNEn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNEn53 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        # self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNDe53 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNDe52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNDe51 = nn.BatchNorm2d(512, momentum=bn_momentum)

        # self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNDe43 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.BNDe42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        # self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.BNDe41 = nn.BatchNorm2d(256, momentum=bn_momentum)

        # self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.BNDe33 = nn.BatchNorm2d(256, momentum=bn_momentum)
        # self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.BNDe32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        # self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.BNDe31 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.ConvDe11 = nn.Conv2d(
            64, self.out_channels, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_channels, momentum=bn_momentum)

    def forward(self, x):

        # ENCODE LAYERS
        # Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        # Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        # # Stage 3
        # x = F.relu(self.BNEn31(self.ConvEn31(x)))
        # x = F.relu(self.BNEn32(self.ConvEn32(x)))
        # x = F.relu(self.BNEn33(self.ConvEn33(x)))
        # x, ind3 = self.MaxEn(x)
        # size3 = x.size()

        # # Stage 4
        # x = F.relu(self.BNEn41(self.ConvEn41(x)))
        # x = F.relu(self.BNEn42(self.ConvEn42(x)))
        # x = F.relu(self.BNEn43(self.ConvEn43(x)))
        # x, ind4 = self.MaxEn(x)
        # size4 = x.size()

        # # Stage 5
        # x = F.relu(self.BNEn51(self.ConvEn51(x)))
        # x = F.relu(self.BNEn52(self.ConvEn52(x)))
        # x = F.relu(self.BNEn53(self.ConvEn53(x)))
        # x, ind5 = self.MaxEn(x)
        # size5 = x.size()

        # # DECODE LAYERS
        # # Stage 5
        # x = self.MaxDe(x, ind5, output_size=size4)
        # x = F.relu(self.BNDe53(self.ConvDe53(x)))
        # x = F.relu(self.BNDe52(self.ConvDe52(x)))
        # x = F.relu(self.BNDe51(self.ConvDe51(x)))

        # # Stage 4
        # x = self.MaxDe(x, ind4, output_size=size3)
        # x = F.relu(self.BNDe43(self.ConvDe43(x)))
        # x = F.relu(self.BNDe42(self.ConvDe42(x)))
        # x = F.relu(self.BNDe41(self.ConvDe41(x)))

        # # Stage 3
        # x = self.MaxDe(x, ind3, output_size=size2)
        # x = F.relu(self.BNDe33(self.ConvDe33(x)))
        # x = F.relu(self.BNDe32(self.ConvDe32(x)))
        # x = F.relu(self.BNDe31(self.ConvDe31(x)))

        # Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        # Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        x = F.softmax(x, dim=1)

        return x


class Train():
    default_hps = {
        "batch_size": 4,
        "epochs": 10,
        "learning_rate": 0.005,
        "sgd_momentum": 0.9,
        "bn_momentum": 0.5,
        "cross_entropy_loss_weights": [1.0, 15.0],
        "no_cuda": False,
        "seed": 42,
        "in_chn": 3,
        "out_chn": 2
    }

    @staticmethod
    def save_checkpoint(state, path):
        torch.save(state, path)
        print("Checkpoint saved at {}".format(path))

    @staticmethod
    # epochs is target epoch, path is provided to load saved checkpoint
    def Train(trainloader, hyperparams=default_hps, path=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = OLEDUNet(bn_momentum=hyperparams['bn_momentum'])
        model.to(device)
        optimizer = optim.SGD(model.parameters(),
                              lr=hyperparams['learning_rate'], momentum=hyperparams['sgd_momentum'])
        loss_fn = nn.CrossEntropyLoss()
        run_epoch = hyperparams['epochs']

        if path == None:
            epoch = 0
            path = os.path.join(os.getcwd(), 'oledunet_weights.pth.tar')
            print("Creating new checkpoint '{}'".format(path))
        else:
            if os.path.isfile(path):
                print("Loading checkpoint '{}'".format(path))
                checkpoint = torch.load(path)
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded checkpoint '{}' (epoch {})".format(
                    path, checkpoint['epoch']))
            else:
                print("No checkpoint found at '{}'".format(path))

        for i in range(1, run_epoch + 1):
            print('Epoch {}:'.format(i))
            sum_loss = 0.0

            for j, data in enumerate(trainloader, 1):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()

                print('Loss at {} mini-batch: {}'.format(j,
                                                         loss.item() / trainloader.batch_size))

            print(
                'Average loss @ epoch: {}'.format((sum_loss / j * trainloader.batch_size)))
            if device == "cuda":
                torch.cuda.empty_cache()

        print("Training complete. Saving checkpoint...")
        Train.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)


class OLEDDataset(Dataset):
    def __init__(self, raw_dir, oled_dir):
        self.raw_dir = raw_dir
        self.oled_dir = oled_dir
        self.raw_ims_paths = [os.path.join(
            raw_dir, file) for file in os.listdir(raw_dir)]
        self.oled_ims_paths = [os.path.join(
            oled_dir, file) for file in os.listdir(oled_dir)]

    def __len__(self):
        return len(self.raw_ims_paths)

    def __getitem__(self, idx):
        image_raw = torchvision.io.read_image(
            self.raw_ims_paths[idx], mode=torchvision.io.ImageReadMode.RGB)
        image_oled = torchvision.io.read_image(
            self.oled_ims_paths[idx], mode=torchvision.io.ImageReadMode.GRAY)

        data = (image_raw.type(torch.float32),
                image_oled.type(torch.float32))
        # print([i.shape for i in data])

        return data
