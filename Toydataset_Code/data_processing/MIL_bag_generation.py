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
    # dataset = 2
    # bag_num = [1000, 200]
    dataset = 2
    bag_num = [1000, 200]
    bag_size = 8
    MIL = 1
    pos_ratio = 0.5

    bag_folder = '/Data3/GCA_Demo/Dataset%d_Bag_%d_%d_%d' % (dataset, bag_num[0],bag_size, MIL)

    data_folder = ['/Data3/GCA_Demo/Dataset%d_demo' % (dataset),'/Data3/GCA_Demo/Dataset%d_demo_val' % (dataset)]

    mode = ['Train','Val']
    r = np.random.RandomState(1)

    for mi in range(len(mode)):
        save_root = os.path.join(bag_folder, mode[mi])
        if not os.path.exists(save_root):
            os.makedirs(save_root)
            
        img_root = data_folder[mi]
        now_bag_num = int(bag_num[mi])

        for bi in range(now_bag_num):
            now_label = np.random.randint(0, 2, 1)[0]
            pos_list = glob.glob(os.path.join(img_root,'1','*size1024.png'))
            neg_list = glob.glob(os.path.join(img_root,'0','*size1024.png'))
    
            if now_label == 0:
                indices = torch.LongTensor(r.randint(0, len(neg_list), bag_size))
                
                if bag_size == 1:
                    now_images = [neg_list[indices]]
                else:
                    now_images = list(np.array(neg_list)[indices])

            else:
                if MIL:
                    pos_length = r.randint(1, int(pos_ratio * bag_size), 1)
                else:
                    pos_length = bag_size
                neg_length = bag_size - pos_length
    
                indices_pos = torch.LongTensor(r.randint(0, len(pos_list), pos_length))
                indices_neg = torch.LongTensor(r.randint(0, len(neg_list), neg_length))
    
                if len(indices_pos) == 1:
                    now_images_pos = [pos_list[indices_pos]]
                    now_images_neg = list(np.array(neg_list)[indices_neg])
                else:
                    now_images_pos = list(np.array(pos_list)[
                                              indices_pos])  # + list(np.array(self.image_list_class0)[indices_neg])   # self.image_list_class0[indices_neg]
                    now_images_neg = list(np.array(neg_list)[indices_neg])
    
                now_images = now_images_pos + now_images_neg
                random.shuffle(now_images)
    

            df = pd.DataFrame(columns = ['img_root', 'class'])
            for ki in range(len(now_images)):
                df.loc[ki] = [now_images[ki], now_label]
                
            df.to_csv(os.path.join(save_root, 'bag_%d.csv' % (bi)), index = False)



