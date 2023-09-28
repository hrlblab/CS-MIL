import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math

import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import glob
import imgaug.augmenters as iaa


import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import glob
from torch.utils.data import DataLoader, random_split
import scipy.ndimage
import cv2
import PIL
import sys


class MILDataSet(data.Dataset):
    def __init__(self, root, crop_size=(512, 512), bag_batch = 16):
        self.root = root
        self.shape_h, self.shape_w = crop_size
        self.batch = bag_batch

        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.ScaleX((0.75, 1.5)),
            iaa.ScaleY((0.75, 1.5))
        ])

        self.image_aug_color = iaa.Sequential([
            # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.GammaContrast((0, 2.0)),
            iaa.Add((-0.1, 0.1), per_channel=0.5),
            #iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), # new
            #iaa.AddToHueAndSaturation((-0.1, 0.1)),
            #iaa.GaussianBlur(sigma=(0, 1.0)), # new
            #iaa.AdditiveGaussianNoise(scale=(0, 0.1)), # new
        ])

        self.image_aug_noise = iaa.Sequential([
            # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            #iaa.GammaContrast((0.5, 2.0)),
            #iaa.Add((-0.1, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),  # new
            # iaa.AddToHueAndSaturation((-0.1, 0.1)),
            iaa.GaussianBlur(sigma=(0, 1.0)),  # new
            iaa.AdditiveGaussianNoise(scale=(0, 0.1)),  # new
        ])

        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        self.image_aug_256 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((-10, 10), per_channel=0.5)
        ])

        case_list = []
        label_list = []
        classes = glob.glob(os.path.join(self.root,'*'))

        for ki in range(len(classes)):
            label = os.path.basename(classes[ki])
            cases = glob.glob(os.path.join(classes[ki],'*'))

            for ci in range(len(cases)):
                case_list.append(cases[ci])
                label_list.append(int(label))

        self.cases = []

        print("listing the cases........")
        for i in range(len(case_list)):
            case_folder = case_list[i]
            label = label_list[i]
            name = osp.basename(case_folder)
            image_list = glob.glob(os.path.join(case_folder,'*.png'))

            self.cases.append({
                "case": case_folder,
                'images': image_list,
                "label": label,
                "name": name,
            })
        print('{} cases are loaded!'.format(len(case_list)))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        datafiles = self.cases[index]
        # read png file

        batch_bag = np.zeros((self.batch, self.shape_h, self.shape_w, 3))
        image_ind = np.random.randint(0, len(datafiles["images"]),size = self.batch)

        for bi in range(len(image_ind)):
            img_ind = int(image_ind[bi])
            now_image = plt.imread(datafiles["images"][img_ind])[:,:,:3]
            batch_bag[bi] = now_image

        seed = np.random.rand(4)
        batch_bag = batch_bag.astype(np.float64).copy()

        batch_bag = (batch_bag * 255).astype(np.uint8)
        # if seed[0] > 0.5:
        #     image, label = self.image_mask_aug(images=image, heatmaps=label)
        if seed[0] > 0.5:
            batch_bag = self.image_mask_aug(images=batch_bag)

        if seed[1] > 0.5:
            batch_bag = self.image_aug_color(images=batch_bag)

        if seed[2] > 0.5:
            batch_bag = self.image_aug_noise(images=batch_bag)

        batch_bag = (batch_bag / 255).astype(np.float32)

        batch_bag = batch_bag.transpose((0,3,1,2))

        label = int(datafiles["label"])
        name = datafiles["name"]
        return batch_bag.copy(), label, name


class MILValDataSet(data.Dataset):
    def __init__(self, root, crop_size=(512, 512), bag_batch = 16):
        self.root = root
        self.shape_h, self.shape_w = crop_size
        self.batch = bag_batch

        case_list = []
        label_list = []
        classes = glob.glob(os.path.join(self.root, '*'))

        for ki in range(len(classes)):
            label = os.path.basename(classes[ki])
            cases = glob.glob(os.path.join(classes[ki], '*'))

            for ci in range(len(cases)):
                case_list.append(cases[ci])
                label_list.append(int(label))

        self.cases = []

        print("listing the cases........")
        for i in range(len(case_list)):
            case_folder = case_list[i]
            label = label_list[i]
            name = osp.basename(case_folder)
            image_list = glob.glob(os.path.join(case_folder, '*.png'))

            self.cases.append({
                "case": case_folder,
                'images': image_list,
                "label": label,
                "name": name,
            })
        print('{} cases are loaded!'.format(len(case_list)))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        datafiles = self.cases[index]
        # read png file

        batch_bag = np.zeros((self.batch, self.shape_h, self.shape_w, 3))
        image_ind = np.random.randint(0, len(datafiles["images"]),size = self.batch)

        for bi in range(len(image_ind)):
            img_ind = int(image_ind[bi])
            now_image = plt.imread(datafiles["images"][img_ind])[:,:,:3]
            batch_bag[bi] = now_image

        batch_bag = batch_bag.astype(np.float32)

        batch_bag = batch_bag.transpose((0,3,1,2))

        label = int(datafiles["label"])
        name = datafiles["name"]
        return batch_bag.copy(), label, name

if __name__ == '__main__':

    trainset_dir = '/Data2/GCA/Multi-instance-learning/data'

    itrs_each_epoch = 250
    batch_size = 1
    input_size = (512,512)

    trainloader = DataLoader(
        MILValDataSet(trainset_dir, crop_size=input_size, bag_batch = 16),batch_size = 1, shuffle = False, num_workers =0)

    for iter, batch in enumerate(trainloader):
        print(iter)
        # imgs = torch.from_numpy(batch['image']).cuda()
        # lbls = torch.from_numpy(batch['label']).cuda()
        # volumeName = batch['name']
        # t_ids = torch.from_numpy(batch['task_id']).cuda()
        # s_ids = torch.from_numpy(batch['scale_id']).cuda()
