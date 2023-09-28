import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import glob

import re

import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from skimage.transform import resize
import torch

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    

if __name__ == "__main__":
    dataset = 1
    bag_size = 8

    crop_patch = 0
    create_bag = 1

    bag_folder = '/Data3/GCA_Demo/4864patch_dataset%d_patch' % (dataset)
    data_folder = '/Data3/GCA_Demo/4864patch_dataset%d' % (dataset)
    bag_csv_folder = '/Data3/GCA_Demo/4864patch_dataset%d_bag' % (dataset)

    mode = ['Val']
    r = np.random.RandomState(1)

    for mi in range(len(mode)):
        save_root = bag_folder
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        imgs = glob.glob(os.path.join(data_folder,'*size1024.png'))

        for im in range(len(imgs)):
        #for im in range(0,5):

            now_img = imgs[im]
            region_name_list = os.path.basename(now_img).split('_')
            region_name = region_name_list[0] + '_' + region_name_list[1] + '_' + region_name_list[2]  + '_' + region_name_list[3]  + '_' + region_name_list[4]  + '_' + region_name_list[5]  + '_' + region_name_list[6]  + '_' + region_name_list[7] +  '_' + region_name_list[8] +  '_' + region_name_list[9]

            patch_output_folder = os.path.join(bag_folder, region_name)

            if not os.path.exists(patch_output_folder):
                os.makedirs(patch_output_folder)

            if not os.path.exists(bag_csv_folder):
                os.makedirs(bag_csv_folder)
            if crop_patch:
                region_256 = plt.imread(now_img.replace('size1024', 'size256'))[:,:,:3]
                region_1024 = plt.imread(now_img)[:,:,:3]
                region_512 = plt.imread(now_img.replace('size1024', 'size512'))[:,:,:3]


                for xi in range(0,16):
                    for yi in range(0,16):
                        now_x = xi * 256 + 384
                        now_y = yi * 256 + 384

                        patch_256 = region_256[now_x:now_x + 256, now_y:now_y + 256, :]

                        if patch_256.min() != 0:
                            continue

                        patch_512 = region_512[now_x - 128:now_x - 128 + 512, now_y - 128:now_y - 128 + 512, :]
                        patch_1024 = region_512[now_x - 384:now_x - 384 + 1024, now_y - 384:now_y - 384 + 1024, :]

                        patch_512 = cv2.resize(patch_512, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)
                        patch_1024 = cv2.resize(patch_1024, (256,256), 0, 0, interpolation = cv2.INTER_NEAREST)

                        plt.imsave(os.path.join(patch_output_folder,'region_%d_%d_size256.png' % (now_x, now_y)), patch_256)
                        plt.imsave(os.path.join(patch_output_folder,'region_%d_%d_size512.png' % (now_x, now_y)), patch_512)
                        plt.imsave(os.path.join(patch_output_folder,'region_%d_%d_size1024.png' % (now_x, now_y)), patch_1024)

            if create_bag:
                for i in range(10):
                    regions = glob.glob(os.path.join(patch_output_folder, '*size1024.png'))
                    if len(regions) == 0:
                        continue
                    random.shuffle(regions)
                    now_label = 0

                    bag_num = int(len(regions) / bag_size) + 1

                    for bi in range(bag_num):
                        if bi != bag_num - 1:
                            bag_start = bi * bag_size
                        else:
                            bag_start = len(regions) - bag_size
                        now_bag = bi
                        df = pd.DataFrame(columns=['img_root', 'class'])
                        for ki in range(8):
                            df.loc[ki] = [regions[bag_start + ki], now_label]
                            df.to_csv(os.path.join(bag_csv_folder, 'bag_%d_%s_%d.csv' % (i, region_name, bi)), index = False)
                #
                #
                # random.shuffle(regions)
                #
                # for bi in range(bag_num):
                #     if bi != bag_num - 1:
                #         bag_start = bi * bag_size
                #     else:
                #         bag_start = len(regions) - 8
                #     now_bag = bi
                #     df = pd.DataFrame(columns=['img_root', 'class'])
                #     for ki in range(8):
                #         df.loc[ki] = [regions[bag_start + ki], now_label]
                #         df.to_csv(os.path.join(bag_csv_folder, 'bag_1_%s_%d.csv' % (region_name, bi)), index = False)
                #
                #
                # random.shuffle(regions)
                #
                # for bi in range(bag_num):
                #     if bi != bag_num - 1:
                #         bag_start = bi * bag_size
                #     else:
                #         bag_start = len(regions) - 8
                #     now_bag = bi
                #     df = pd.DataFrame(columns=['img_root', 'class'])
                #     for ki in range(8):
                #         df.loc[ki] = [regions[bag_start + ki], now_label]
                #         df.to_csv(os.path.join(bag_csv_folder, 'bag_2_%s_%d.csv' % (region_name, bi)), index = False)