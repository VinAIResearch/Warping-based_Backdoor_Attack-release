import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    else:
        raise Exception("Invalid dataset")
    return dataset


def main():
    pass


if __name__ == "__main__":
    main()
