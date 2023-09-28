from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

data_dir = '/Data2/GCA/AttentionDeepMIL-master/data/train'
output_dir = '/Data2/GCA/AttentionDeepMIL-master/contrastive_learning_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

label = glob.glob(os.path.join(data_dir, "*"))

for ki in range(len(label)):
    cases = glob.glob(os.path.join(label[ki], "*"))

    # df = pd.DataFrame(columns = ['root', 'label', 'cluster'])

    for now_case in cases:
        images = glob.glob(os.path.join(now_case,"*"))
        images.sort()

        for now_image in images:
            patch = plt.imread(now_image)[:,:,:3]
            image_name = os.path.basename(now_image)
            if (patch.mean(2) > 230 / 255).sum() < 512 * 512 / 2:  # for dodnet
                plt.imsave(os.path.join(output_dir, image_name), patch)

