from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

severe_list = [3]
resolution_list = [256,512,1024]

for sl in range(len(severe_list)):
    severe = severe_list[sl]
    for rl in range(len(resolution_list)):
        resolution = resolution_list[rl]

        print('now is : severe %d, resolution %d' % (severe, resolution))

        if severe == 0:
            feature_dir = '/Data2/GCA/simsiam/feature_data_multi-resolution/'
            csv_dir = '/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_oneclustering_only%d' % (resolution)
            label_file = pd.read_csv('/Data2/GCA/simsiam/data_list.csv')
        elif severe == 1:
            feature_dir = '/Data2/GCA/simsiam/feature_data_multi-resolution_severe/'
            csv_dir = '/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_severe_oneclustering_only%d' % (resolution)
            label_file = pd.read_csv('/Data2/GCA/data_list_severe.csv')
        elif severe == 2:
            feature_dir = '/Data2/GCA/simsiam/feature_data_multi-resolution_severe/'
            csv_dir = '/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_CD_oneclustering_only%d' % (resolution)
            label_file = pd.read_csv('/Data2/GCA/data_list_CD.csv')
        else:
            feature_dir = '/Data2/GCA/simsiam/feature_data_multi-resolution_severe_extend/'
            csv_dir = '/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_severe_extend_oneclustering_only%d' % (resolution)
            label_file = pd.read_csv('/Data2/GCA/data_list_severe_extend.csv')

        cases = glob.glob(os.path.join(feature_dir, "*"))

        mapping = 0


        for now_case in cases:
        # for now_case in cases[:5]:
            print(now_case)

            now_wsi = os.path.basename(now_case)
            csv_folder = csv_dir

            if len(label_file[label_file['filename'] == now_wsi + '.svs']['class'].tolist()) == 0:
                continue

            now_label = label_file[label_file['filename'] == now_wsi + '.svs']['class'].tolist()[0]

            if not os.path.exists(csv_folder):
                os.makedirs(csv_folder)

            feature_files = glob.glob(os.path.join(now_case,"*size%d.npy" % (resolution)))

            features = np.zeros((len(feature_files), 2048))

            for ii in range(len(feature_files)):
                features[ii] = np.load(feature_files[ii])

            local_features = features
            local_features_file = feature_files

            kmeans = KMeans(n_clusters=8, random_state=0).fit(local_features)
            df = pd.DataFrame(columns=['wsi_name', 'root', 'label', 'cluster'])
            for ii in range(len(local_features)):
                now_root = local_features_file[ii]
                now_wsi = os.path.basename(now_root).split('_')[0]
                now_label = label_file[label_file['filename'] == now_wsi + '.svs']['class'].tolist()[0]

                now_cluster = kmeans.labels_[ii]
                row = len(df)
                df.loc[row] = [now_wsi, now_root, now_label, now_cluster]


            now_wsi = os.path.basename(now_case)
            csv_folder = csv_dir
            if len(label_file[label_file['filename'] == now_wsi + '.svs']['class'].tolist()) == 0:
                continue

            now_label = label_file[label_file['filename'] == now_wsi + '.svs']['class'].tolist()[0]

            save_root = os.path.join(csv_folder, '%s_cluster.csv' % (now_wsi))

            now_df = df[df['wsi_name'] == now_wsi]
            now_df.to_csv(save_root, index = False)

 
