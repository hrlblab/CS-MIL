"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import glob
import os
import imgaug.augmenters as iaa


import matplotlib.pyplot as plt


class MILBags(data_utils.Dataset):
    def __init__(self, root, target_class=1, size = (512, 512), mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.root = root
        self.size = size
        self.target_class = target_class
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.images_list = []
        self.label_list = []

        'augmentation'
        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.ScaleX((0.75, 1.5)),
            iaa.ScaleY((0.75, 1.5))
        ])

        self.image_aug_color = iaa.Sequential([
            iaa.GammaContrast((0, 2.0)),
            iaa.Add((-0.1, 0.1), per_channel=0.5),
        ])

        self.image_aug_noise = iaa.Sequential([

            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),  # new
            iaa.GaussianBlur(sigma=(0, 1.0)),  # new
            iaa.AdditiveGaussianNoise(scale=(0, 0.1)),  # new
        ])

        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        self.image_aug_256 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((-10, 10), per_channel=0.5)
        ])


        self.images_neg_list = []
        self.label_neg_list = []


        'check the dataset size'
        classes = glob.glob(os.path.join(self.root, '*'))
        for ki in range(len(classes)):
            label = os.path.basename(classes[ki])
            cases = glob.glob(os.path.join(classes[ki], '*'))

            for ci in range(len(cases)):
                images = glob.glob(os.path.join(cases[ci], '*'))

                self.images_list.extend((images))
                self.label_list.extend(([label] * len(images)))

                if label == '0':
                    self.images_neg_list.extend((images))
                    self.label_neg_list.extend(([label] * len(images)))

        self.num_in_dataset = len(self.images_list)
        self.num_in_neg_dataset = len(self.images_neg_list)
        self.bags_list, self.labels_list = self._create_bags()

    def _create_bags(self):

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            seed = np.random.rand(1)

            if seed > 0.5:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1

                indices = torch.LongTensor(self.r.randint(0, self.num_in_dataset, bag_length))
                labels_in_bag = np.array(self.label_list)[indices].astype(np.uint8)
                images_in_bag = list(np.array(self.images_list)[indices])
                # labels_in_bag = np.array(labels_in_bag) == self.target_class

                bags_list.append(images_in_bag)
                labels_list.append(labels_in_bag)

            else:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length < 1:
                    bag_length = 1

                indices = torch.LongTensor(self.r.randint(0, self.num_in_neg_dataset, bag_length))
                labels_in_bag = np.array(self.label_neg_list)[indices].astype(np.uint8)
                images_in_bag = list(np.array(self.images_neg_list)[indices])
                # labels_in_bag = np.array(labels_in_bag) == self.target_class

                bags_list.append(images_in_bag)
                labels_list.append(labels_in_bag)
        return bags_list, labels_list

    def __len__(self):
        return len(self.bags_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = [max(self.labels_list[index]), self.labels_list[index]]

        bag_image = np.zeros((len(bag), int(self.size[0]), int(self.size[1]), 3))

        for bi in range(len(bag)):
            bag_image[bi] = plt.imread(bag[bi])[:,:,:3]

        if self.train:
            seed = np.random.rand(4)

            bag_image = (bag_image * 255).astype(np.uint8)
            # if seed[0] > 0.5:
            #     image, label = self.image_mask_aug(images=image, heatmaps=label)
            if seed[0] > 0.5:
                bag_image = self.image_mask_aug(images=bag_image)

            if seed[1] > 0.5:
                bag_image = self.image_aug_color(images=bag_image)

            if seed[2] > 0.5:
                bag_image = self.image_aug_noise(images=bag_image)

            bag_image = (bag_image / 255).astype(np.float32)

        bag_image = bag_image.transpose((0, 3, 1, 2))

        return bag_image, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MILBags(root = 'data/train',
                                                   target_class=1,
                                                   size = (512, 512),
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=1000,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MILBags(root = 'data/test',
                                                  target_class=1,
                                                  size=(512, 512),
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=250,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
