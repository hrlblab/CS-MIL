"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import glob
import os
import imgaug.augmenters as iaa
import pandas as pd
import random

import matplotlib.pyplot as plt
# import imgaug.augmenters as iaa


class MILBags(data_utils.Dataset):
    def __init__(self, data_root, size = (256, 256), mean_bag_length = 64, var_bag_length = 4, seed=1, train=True, bag_num = 200, pos_ratio = 0.5, multi_scale = True, MIL = True):
        self.data_root = data_root
        self.size = size
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.train = train
        self.pos_ratio = pos_ratio
        self.multi_scale = multi_scale
        self.MIL = MIL

        self.r = np.random.RandomState(seed)

        self.image_list_class1 = glob.glob(os.path.join(self.data_root,"1",'*size1024.png'))
        self.image_list_class0 = glob.glob(os.path.join(self.data_root,"0",'*size1024.png'))

        self.bag_num = bag_num

    def _create_bags(self, index):

        bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1

        now_label = np.random.randint(0,2,1)
        if now_label == 0:
            now_images = self.image_list_class0
            indices = torch.LongTensor(self.r.randint(0, len(now_images), bag_length))
            images_bags = np.zeros((bag_length, 3, 256, 256, 3))
            

            for ii in range(bag_length):
                if self.multi_scale:
                    images_bags[ii, 0, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size256'))[:, :,
                                              :3]
                    images_bags[ii, 1, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size512'))[:, :,
                                              :3]
                    images_bags[ii, 2, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size1024'))[:, :,
                                              :3]
                    if bag_length == 1:
                        now_images = [self.image_list_class0[indices]]
                    else:
                        now_images = list(np.array(self.image_list_class0)[indices])
                else:
                    images_bags[ii, 0, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size256'))[:, :, :3]
                    images_bags[ii, 1, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size256'))[:, :, :3]
                    images_bags[ii, 2, ...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size256'))[:, :, :3]

                    if bag_length == 1:
                        now_images = [self.image_list_class0[indices]]
                    else:
                        now_images = list(np.array(self.image_list_class0)[indices])
        else:
            if self.MIL:
                pos_length = self.r.randint(0, int(self.pos_ratio * bag_length), 1)
            else:
                pos_length = bag_length
            neg_length = bag_length - pos_length

            indices_pos = torch.LongTensor(self.r.randint(0, len(self.image_list_class1), pos_length))
            indices_neg = torch.LongTensor(self.r.randint(0, len(self.image_list_class0), neg_length))

            if len(indices_pos) == 1:
                now_images_pos = [self.image_list_class1[indices_pos]]
                now_images_neg = list(np.array(self.image_list_class0)[indices_neg])
            else:
                now_images_pos = list(np.array(self.image_list_class1)[indices_pos])  #+ list(np.array(self.image_list_class0)[indices_neg])   # self.image_list_class0[indices_neg]
                now_images_neg = list(np.array(self.image_list_class0)[indices_neg])

            now_images = now_images_pos + now_images_neg
            random.shuffle(now_images)

            images_bags = np.zeros((bag_length, 3, 256, 256, 3))

            if bag_length == 1:
                if self.multi_scale:
                    images_bags[0, 0, ...] = plt.imread(now_images[0].replace('size1024', 'size256'))[:, :, :3]
                    images_bags[ii, 1, ...] = plt.imread(now_images[0].replace('size1024', 'size512'))[:, :, :3]
                    images_bags[ii, 2, ...] = plt.imread(now_images[0].replace('size1024', 'size1024'))[:, :, :3]

                else:
                    images_bags[0, 0, ...] = plt.imread(now_images[0].replace('size1024', 'size256'))[:, :, :3]
                    images_bags[0, 1, ...] = plt.imread(now_images[0].replace('size1024', 'size256'))[:, :, :3]
                    images_bags[0, 2, ...] = plt.imread(now_images[0].replace('size1024', 'size256'))[:, :, :3]

            else:
                for ii in range(len(now_images)):
                    if self.multi_scale:
                        images_bags[ii, 0, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
                        images_bags[ii, 1, ...] = plt.imread(now_images[ii].replace('size1024', 'size512'))[:, :, :3]
                        images_bags[ii, 2, ...] = plt.imread(now_images[ii].replace('size1024', 'size1024'))[:, :, :3]

                    else:
                        images_bags[ii, 0, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
                        images_bags[ii, 1, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
                        images_bags[ii, 2, ...] = plt.imread(now_images[ii].replace('size1024', 'size256'))[:, :, :3]
        label_bag = now_label
        mask = 1
        #print(now_images)

        # if now_label == 0:
        #     images_bags = images_bags * 0
        return images_bags, label_bag, mask, now_images

    def __len__(self):
        return self.bag_num

    def __getitem__(self, index):
        bag, label, mask, root = self._create_bags(index)
        return bag, label, mask, root

if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MILBags(data_root = '/Data2/GCA_Demo/Dataset1_demo',
                                                   size = (256, 256),
                                                   mean_bag_length=64,
                                                   var_bag_length=4,
                                                   seed=1,
                                                   train=1),
                                         batch_size=1,
                                         shuffle=True)

    validation_loader = data_utils.DataLoader(MILBags(data_root = '/Data2/GCA_Demo/Dataset1_demo_val',
                                                  size=(256, 256),
                                                  mean_bag_length=64,
                                                  var_bag_length=4,
                                                  seed=1,
                                                  train=2),
                                        batch_size=1,
                                        shuffle=False)


    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label, mask, case_name) in enumerate(train_loader):
        print('aaa')
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))

