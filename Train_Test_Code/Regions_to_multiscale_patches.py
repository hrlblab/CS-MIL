import os

import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from skimage.transform import resize

if __name__ == "__main__":
    folder_name = '/Data2/CSMIL_TCGA/TCGA_regions'
    output_folder =  '/Data2/CSMIL_TCGA/TCGA_multi-scale_patches'

    cases = glob.glob(os.path.join(folder_name,'*'))

    for case in cases:
        regions = glob.glob(os.path.join(case, '*.png'))

        now_folder = os.path.join(output_folder, os.path.basename(case))

        if not os.path.exists(now_folder):
            os.makedirs(now_folder)

        for ri in range(len(regions)):
            now_img = plt.imread(regions[ri])[:,:,:3]
            for xi in range(0,16):
                for yi in range(0,16):
                    now_x = xi * 256 + 384
                    now_y = yi * 256 + 384

                    patch_256 = now_img[now_x:now_x + 256, now_y:now_y + 256,:]
                    patch_512 = resize(now_img[now_x - 128:now_x + 256 + 128, now_y - 128:now_y + 256 + 128,:], (256, 256,3), anti_aliasing= False)
                    patch_1024 = resize(now_img[now_x - 128 -256:now_x + 256 + 128 +256, now_y - 128 - 256:now_y + 256 + 128 + 256,:], (256, 256,3), anti_aliasing= False)

                    plt.imsave(os.path.join(now_folder, os.path.basename(regions[ri]).replace('size4864.png', '%d_%d_size256.png' % (now_x, now_y))),patch_256)
                    plt.imsave(os.path.join(now_folder, os.path.basename(regions[ri]).replace('size4864.png', '%d_%d_size512.png' % (now_x, now_y))),patch_512)
                    plt.imsave(os.path.join(now_folder, os.path.basename(regions[ri]).replace('size4864.png', '%d_%d_size1024.png' % (now_x, now_y))),patch_1024)














