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
    def __init__(self, data_csv_root, feature_csv_root, size = (256, 256), mean_bag_length = 64, var_bag_length = 4, seed=1, train=True):
        self.feature_csv_root = feature_csv_root
        self.data_csv_root = data_csv_root
        self.size = size
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.train = train

        self.r = np.random.RandomState(seed)
        self.data_csv = pd.read_csv(data_csv_root)

        self.image_list = []
        self.label_list = []

#        self.data_aug =


        for ki in range(len(self.data_csv)):
            now_case, now_label, train = self.data_csv.iloc[ki]['filename'], self.data_csv.iloc[ki]['class'], self.data_csv.iloc[ki]['train']
            if (self.train == 1) and (train == 1):
                self.image_list.append((pd.read_csv(os.path.join(self.feature_csv_root,now_case.replace('.svs', '.csv')))))
                self.label_list.append((now_label))

            elif (self.train == 2) and (train == 2):
                self.image_list.append((pd.read_csv(os.path.join(self.feature_csv_root,now_case.replace('.svs', '.csv')))))
                self.label_list.append((now_label))

        self.case_num = len(self.image_list)

    def _create_bags(self, index):
        now_images = self.image_list[index]['root'].tolist()
        now_label = self.label_list[index]

        bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1

        indices = torch.LongTensor(self.r.randint(0, len(now_images), bag_length))
        images_bags = np.zeros((bag_length, 3, 256, 256, 3))

        for ii in range(len(indices)):
            images_bags[ii,2,...] = plt.imread(now_images[indices[ii]])[:,:,:3]
            images_bags[ii,1,...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size512'))[:,:,:3]
            images_bags[ii,0,...] = plt.imread(now_images[indices[ii]].replace('size1024', 'size256'))[:,:,:3]

        # if now_label == 0:
        #     images_bags = images_bags * 0

        label_bag = now_label
        mask = 1
        case_name = os.path.basename(now_images[0]).split('_')[0]

        return images_bags, label_bag, mask, case_name

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        bag, label, mask, case_name = self._create_bags(index)
        return bag, label, mask, case_name


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MILBags(data_csv_root = '/Data2/GCA_Demo/data_list_severe.csv',
                                                 feature_csv_root = '/Data2/GCA_Demo/Datase1_csv',
                                                   size = (256, 256),
                                                   mean_bag_length=64,
                                                   var_bag_length=4,
                                                   seed=1,
                                                   train=1),
                                         batch_size=1,
                                         shuffle=True)

    validation_loader = data_utils.DataLoader(MILBags(data_csv_root = '/Data2/GCA_Demo/data_list_severe.csv',
                                                 feature_csv_root = '/Data2/GCA_Demo/Datase1_csv',
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
    #     mnist_bags_train += label[0].numpy()[0]
    # print('Number positive train bags: {}/{}\n'
    #       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
    #     mnist_bags_train, len(train_loader),
    #     np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

