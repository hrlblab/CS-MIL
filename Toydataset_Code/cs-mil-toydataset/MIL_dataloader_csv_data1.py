"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import glob
import os
#import imgaug.augmenters as iaa
import pandas as pd
import random

import matplotlib.pyplot as plt
# import imgaug.augmenters as iaa


class MILBags(data_utils.Dataset):
    def __init__(self, bag_root, MIL):
        self.bag_root = bag_root
        self.bag_list = glob.glob(os.path.join(self.bag_root,'*'))
        self.MIL = MIL

    def _create_bags(self, index):

        df = pd.read_csv(self.bag_list[index])
        now_images = df['img_root'].tolist()

        images_bags = np.zeros((len(df), 3, 256, 256, 3))
        for ii in range(len(df)):
            if self.MIL:
                images_bags[ii, 0, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :,:3]
                images_bags[ii, 1, ...] = plt.imread(now_images[ii].replace('size1024', 'size512'))[:, :,:3]
                images_bags[ii, 2, ...] = plt.imread(now_images[ii].replace('size1024', 'size1024'))[:, :,:3]

            else:
                images_bags[ii, 0, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
                images_bags[ii, 1, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
                images_bags[ii, 2, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3] 


        label_bag = int(df['class'].tolist()[0])
        mask = 1
        #print(now_images)

        # if label_bag == 0:
        #     images_bags = images_bags * 0
        return images_bags, label_bag, mask, now_images

    def __len__(self):
        return len(self.bag_list)

    def __getitem__(self, index):
        bag, label, mask, root = self._create_bags(index)
        return bag, label, mask, root

if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MILBags(bag_root = '/Data2/GCA_Demo/Bag_1000_1/Train',
                                                  MIL = False),
                                         batch_size=1,
                                         shuffle=True)

    validation_loader = data_utils.DataLoader(MILBags(bag_root = '/Data2/GCA_Demo/Bag_1000_1/Train',
                                                  MIL = False),
                                        batch_size=1,
                                        shuffle=False)


    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label, mask, case_name) in enumerate(train_loader):
        print('aaa')
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))

