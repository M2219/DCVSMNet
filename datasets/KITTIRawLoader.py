import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np


IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def ktraw_loader(filepath):

    left_path = os.path.join(filepath, 'image_02/data')
    right_path = os.path.join(filepath, 'image_03/data')

    total_name = [name for name in os.listdir(left_path)]

    val_left = []
    val_right = []
    for name in total_name:
        val_left.append(os.path.join(left_path, name))
        val_right.append(os.path.join(right_path, name))

    return val_left, val_right


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class myDataset(data.Dataset):

    def __init__(self, left, right, imgloader=img_loader):
        self.left = left
        self.right = right

        self.imgloader = imgloader

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]

        limg = self.imgloader(left)
        rimg = self.imgloader(right)

        w, h = limg.size

        limg = limg.crop((w-1242, h-375, w, h))
        rimg = rimg.crop((w-1242, h-375, w, h))

        limg = self.transform(limg)
        rimg = self.transform(rimg)

        return limg, rimg

    def __len__(self):
        return len(self.left)

