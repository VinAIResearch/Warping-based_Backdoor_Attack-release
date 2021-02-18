import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Module
from torchvision import transforms

from .blocks import *


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def __call__(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x


# ---------------------------- AE ----------------------------#
class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.downsample1 = Conv2dBlock(3, 12, 3, 2, 1, batch_norm=True, relu=True)
        self.downsample2 = Conv2dBlock(12, 24, 3, 2, 1, batch_norm=True, relu=True)
        self.downsample3 = Conv2dBlock(24, 48, 3, 2, 1, batch_norm=True, relu=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample1 = ConvTranspose2dBlock(48, 24, 4, 2, 1, batch_norm=True, relu=True)
        self.upsample2 = ConvTranspose2dBlock(24, 12, 4, 2, 1, batch_norm=True, relu=True)
        self.upsample3 = ConvTranspose2dBlock(12, 3, 4, 2, 1, batch_norm=True, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AE(Module):
    def __init__(self, opt):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.normalizer = self._get_normalizer(opt)
        self.denormalizer = self._get_denormalizer(opt)

    def forward(self, x):
        x = self.decoder(self.encoder(x))
        if self.normalizer:
            x = self.normalizer(x)
        return x

    def _get_denormalizer(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalizer(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer


class GridGenerator(Module):
    def __init__(self):
        super(GridGenerator, self).__init__()
        self.downsample = Encoder()
        # self.flatten = nn.Flatten()
        self.conv1 = Conv2dBlock(48, 24, 3, 1, 1, batch_norm=False, relu=True)
        self.conv2 = Conv2dBlock(24, 2, 3, 1, 1, batch_norm=False, relu=False)

        # self.linear1 = nn.Linear(48 * 4 * 4, 24 * 4 * 4)
        # self.linear2 = nn.Linear(24 * 4 * 4, 2 * 8 * 8)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.downsample(x)
        # x = self.flatten(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = F.upsample(x, scale_factor=8, mode="bicubic").permute(0, 2, 3, 1)
        return x


class NoiseGenerator(nn.Sequential):
    def __init__(self, opt, in_channels=8, steps=3, channel_init=128):
        super(NoiseGenerator, self).__init__()
        self.steps = steps
        channel_current = in_channels
        channel_next = channel_init
        for step in range(steps):
            self.add_module("upsample_{}".format(step), nn.Upsample(scale_factor=(2, 2), mode="bilinear"))
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
        self.add_module("convblock_up_{}".format(2 * steps), Conv2dBlock(channel_current, 2, relu=False))

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x)
        return x


# ---------------------------- Classifiers ----------------------------#

nclasses = 43  # GTSRB as 43 classes


class NetC_GTRSB(nn.Module):
    def __init__(self):
        super(NetC_GTRSB, self).__init__()

        self.block1 = Conv2dBlock(3, 32)
        self.block2 = Conv2dBlock(32, 32)
        self.downsample1 = DownSampleBlock(p=0.3)

        self.block3 = Conv2dBlock(32, 64)
        self.block4 = Conv2dBlock(64, 64)
        self.downsample2 = DownSampleBlock(p=0.3)

        self.block5 = Conv2dBlock(64, 128)
        self.block6 = Conv2dBlock(128, 128)
        self.downsample3 = DownSampleBlock(p=0.3)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4 * 4 * 128, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.linear11 = nn.Linear(512, nclasses)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


# class NetC_MNIST(nn.Module):
#     def __init__(self):
#         super(NetC_MNIST, self).__init__()
#         self.conv2d_1 = nn.Conv2d(1, 16, (3, 3), 2, 1)
#         self.relu_2 = nn.ReLU(inplace=True)
#         self.dropout_3 = nn.Dropout(0.3)
#         self.conv2d_4 = nn.Conv2d(16, 32, (3, 3), 2, 1)
#         self.relu_5 = nn.ReLU(inplace=True)
#         self.dropout_6 = nn.Dropout(0.3)
#         self.flatten = nn.Flatten()
#         self.linear_8 = nn.Linear(7 * 7 * 32, 128)
#         self.dropout_9 = nn.Dropout(0.3)
#         self.linear_10 = nn.Linear(128, 10)

#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         return x


# class NetC_MNIST(nn.Module):
# def __init__(self):
# super(NetC_MNIST, self).__init__()
# self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
# self.relu2 = nn.ReLU(inplace=True)
# self.dropout3 = nn.Dropout(0.1)

# self.maxpool4 = nn.MaxPool2d((2, 2))
# self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
# self.relu6 = nn.ReLU(inplace=True)
# self.dropout7 = nn.Dropout(0.1)

# self.maxpool5 = nn.MaxPool2d((2, 2))
# self.flatten = nn.Flatten()
# self.linear6 = nn.Linear(64 * 4 * 4, 512)
# self.relu7 = nn.ReLU(inplace=True)
# self.dropout8 = nn.Dropout(0.1)
# self.linear9 = nn.Linear(512, 10)

# def forward(self, x):
# for module in self.children():
# x = module(x)
# return x


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)  # 24
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), 2, 1)  # 12
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 64, (5, 5), 1, 0)  # 8
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), 2, 1)  # 4
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class NetC_CelebA(nn.Module):
    def __init__(self):
        super(NetC_CelebA, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, (3, 3), 1, 1)
        self.backnorm_2 = nn.BatchNorm2d(32)
        self.relu_3 = nn.ReLU(inplace=True)
        self.dropout_4 = nn.Dropout(0.3)

        self.maxpool_5 = nn.MaxPool2d((2, 2))

        self.conv2d_6 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.batchnorm_7 = nn.BatchNorm2d(64)
        self.relu_8 = nn.ReLU(inplace=True)
        self.dropout_9 = nn.Dropout(0.3)

        self.maxpool_11 = nn.MaxPool2d((2, 2))

        self.conv2d_13 = nn.Conv2d(64, 64, (3, 3), 1, 1)
        self.backnorm_14 = nn.BatchNorm2d(64)
        self.relu_15 = nn.ReLU(inplace=True)
        self.dropout_16 = nn.Dropout(0.3)

        self.maxpool_17 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.linear_12 = nn.Linear(64 * 64, 128)
        self.dropout_5 = nn.Dropout(0.3)
        self.linear_13 = nn.Linear(128, 8)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class NetC_CelebA1(nn.Module):
    def __init__(self):
        super(NetC_CelebA1, self).__init__()
        self = torchvision.models.resnet18(pretrained=False)
        self.fc = nn.Linear(512, 8)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def main():
    encoder = GridGenerator()
    a = torch.rand((1, 3, 32, 32))
    a = encoder(a)


if __name__ == "__main__":
    main()


class MNISTBlock3(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class NetC_MNIST3(nn.Module):
    def __init__(self):
        super(NetC_MNIST3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock3(32, 64, 2)  # 7
        self.layer3 = MNISTBlock3(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
