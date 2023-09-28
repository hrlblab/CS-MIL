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


class MILBags(data_utils.Dataset):
    def __init__(self, data_csv_root, feature_csv_root, cluster_num, cluster_list, size = (256, 256), mean_bag_length = 64, var_bag_length = 4, seed=1, train=True):
        self.feature_csv_root = feature_csv_root
        self.data_csv_root = data_csv_root
        self.size = size
        self.cluster_target_list = cluster_list
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.train = train
        self.cluster_num = cluster_num

        self.r = np.random.RandomState(seed)
        self.data_csv = pd.read_csv(data_csv_root)

        self.case_list = []
        self.cluster_list = []
        self.label_list = []
        self.clustercnt_list = []

        def count_cluser(df, cluster_num):
            list = []
            cluster_cnt = np.zeros((cluster_num))
            for ci in range(cluster_num):
                cluster_df = df[df['cluster'] == ci]
                cluster_cnt[ci] = len(cluster_df)
                list.append((cluster_df))

            return list, cluster_cnt


        for ki in range(len(self.data_csv)):
            now_case, now_label, train = self.data_csv.iloc[ki]['filename'], self.data_csv.iloc[ki]['class'], self.data_csv.iloc[ki]['train']
            if (self.train == 1) and (train == 1):
                # now_df = pd.read_csv(os.path.join(self.feature_csv_root,now_case.replace('.svs', '_cluster.csv')))
                now_df = pd.read_csv(glob.glob(os.path.join(self.feature_csv_root,now_case + '_cluster.csv'))[0])
                now_cluster_df, now_cluster_cnt = count_cluser(now_df, self.cluster_num)

                self.case_list.append((now_df))
                self.label_list.append((now_label))
                self.clustercnt_list.append((now_cluster_cnt))
                self.cluster_list.append((now_cluster_df))

            elif (self.train == 2) and (train == 2):
                # now_df = pd.read_csv(os.path.join(self.feature_csv_root, now_case.replace('.svs', '_cluster.csv')))
                now_df = pd.read_csv(glob.glob(os.path.join(self.feature_csv_root,now_case + '_cluster.csv'))[0])
                now_cluster_df, now_cluster_cnt = count_cluser(now_df, self.cluster_num)

                self.case_list.append((now_df))
                self.label_list.append((now_label))
                self.clustercnt_list.append((now_cluster_cnt))
                self.cluster_list.append((now_cluster_df))

            elif (self.train == 0) and (train == 0):
                # now_df = pd.read_csv(os.path.join(self.feature_csv_root, now_case.replace('.svs', '_cluster.csv')))
                now_df = pd.read_csv(glob.glob(os.path.join(self.feature_csv_root,now_case + '_cluster.csv'))[0])
                now_cluster_df, now_cluster_cnt = count_cluser(now_df, self.cluster_num)

                self.case_list.append((now_df))
                self.label_list.append((now_label))
                self.clustercnt_list.append((now_cluster_cnt))
                self.cluster_list.append((now_cluster_df))

        self.case_num = len(self.case_list)

    def _create_bags(self, index):
        non_zero_idx = np.nonzero(self.clustercnt_list[index] > 0)[0][0]

        cluster_cnt = self.clustercnt_list[index]
        now_case = self.cluster_list[index]
        now_label = int(self.label_list[index])
        case_name = now_case[non_zero_idx]['wsi_name'].tolist()[0]

        bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
        if bag_length < 1:
            bag_length = 1

        # cluster_pickup = np.ceil(cluster_cnt / cluster_cnt.sum() * bag_length).astype(int)
        cluster_pickup = np.zeros(len(cluster_cnt)).astype(int)

        for ci in range(len(cluster_pickup)):
            if not ci in self.cluster_target_list:
                cluster_pickup[ci] = 0
            else:
                cluster_pickup[ci] = np.minimum(bag_length, cluster_cnt[ci].astype(int)).astype(int)


        if cluster_pickup.sum() == 0:
            return cluster_pickup, cluster_pickup, cluster_pickup, cluster_pickup, 1

        else:
            addone_cluster_pickup = cluster_pickup.copy()
            addone_cluster_pickup[addone_cluster_pickup == 0] = 1
            mask = np.ones((addone_cluster_pickup.sum()))
            cluster_bag = []
            cluster_label = []
            cnt = 0
            for cl in range(len(cluster_pickup)):
                pickup_num = cluster_pickup[cl]
                if pickup_num == 0:
                    features = np.zeros((1, 2048, 3, 1), dtype=np.float32)
                    mask[cnt] = 0
                    cnt = cnt + 1
                    labels = [now_label]
                else:
                    now_cluster = now_case[cl]['root'].tolist()
                    indices = torch.LongTensor(self.r.randint(0, len(now_cluster), pickup_num))
                    if len(indices) > 1:
                        features_root = list(np.array(now_cluster)[indices])
                        random.shuffle(features_root)
                    else:
                        features_root = [now_cluster[indices]]

                    features = np.zeros((pickup_num, 2048, 3, 1,))

                    for fi in range(pickup_num):
                        '256'
                        features[fi,:,0,:] = np.expand_dims(np.load(features_root[fi]), 1)
                        features[fi,:,1,:] = np.expand_dims(np.load(features_root[fi].replace('size256', 'size512')), 1)
                        features[fi,:,2,:] = np.expand_dims(np.load(features_root[fi].replace('size256', 'size1024')), 1)

                        cnt = cnt + 1
                    labels = [now_label] * pickup_num

                cluster_bag.append((features))
                cluster_label.append((labels))

        return cluster_bag, [now_label, cluster_label], mask, case_name, 0

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        bag_list = []
        label_list = []
        mask_list = []

        for ki in range(200): # before was 100
            bag, label, mask, case_name, skip = self._create_bags(index)
            if skip == 1:
                return bag_list, label_list, mask_list, case_name, 1
            bag_list.append((bag))
            label_list.append((label))
            mask_list.append((mask))

        return bag_list, label_list, mask_list, case_name, 0

if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MILBags(data_csv_root = '/Data2/GCA/AttentionDeepMIL-Resnet50/data_split/data_list_0.csv',
                                                 feature_csv_root = '/Data2/GCA/simsiam/feature_cluster_csv',
                                                 cluster_num = 8,
                                                   size = (256, 256),
                                                   mean_bag_length=64,
                                                   var_bag_length=4,
                                                   seed=1,
                                                   train=0),
                                         batch_size=1,
                                         shuffle=True)

    validation_loader = data_utils.DataLoader(MILBags(data_csv_root = '/Data2/GCA/AttentionDeepMIL-Resnet50/data_split/data_list_0.csv',
                                                feature_csv_root = '/Data2/GCA/simsiam/feature_cluster_csv',
                                                cluster_num = 8,
                                                  size=(256, 256),
                                                  mean_bag_length=64,
                                                  var_bag_length=4,
                                                  seed=1,
                                                  train=1),
                                        batch_size=1,
                                        shuffle=False)

    Testing_loader = data_utils.DataLoader(MILBags(data_csv_root = '/Data2/GCA/AttentionDeepMIL-Resnet50/data_split/data_list_0.csv',
                                                feature_csv_root = '/Data2/GCA/simsiam/feature_cluster_csv',
                                                cluster_num = 8,
                                                  size=(256, 256),
                                                  mean_bag_length=64,
                                                  var_bag_length=4,
                                                  seed=1,
                                                  train=2),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label, mask) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label, mask) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
