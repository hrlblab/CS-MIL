from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

data_dir = '/Data2/GCA/AttentionDeepMIL-master/data/train/'
csv_dir = '/Data2/GCA/AttentionDeepMIL-master/data/csv/'

label = glob.glob(os.path.join(data_dir, "*"))

for ki in range(len(label)):
    cases = glob.glob(os.path.join(data_dir, "*"))

    df = pd.DataFrame(columns = ['root', 'label', 'cluster'])

    for now_case in cases:
        images = glob.glob(os.path.join(now_case,"*"))
        images.sort()

        features = np.zeros((len(images), 1024))

        #for ii in range(len(images)):
        for ii in range(10):
            array[ii] = plt.imread(images[ii])[:,:,:3]

        kmeans = KMeans(n_clusters=8, random_state=0).fit(array)
