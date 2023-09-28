from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

data_dir = '/Data2/GCA/AttentionDeepMIL/Normal_control_vs_patient.csv'
csv_dir = '/Data2/GCA/AttentionDeepMIL/data_list.csv'

label_df = pd.read_csv(data_dir)
list_df = pd.DataFrame(columns = ['filename', 'region', 'patient_type', 'score', 'class', 'train'])

for ki in range(len(label_df)):
    #if label_df.iloc[ki]['patient_type'].replace(" ", "") == 'CD' or label_df.iloc[ki]['patient_type'].replace(" ", "") == 'Control':
    if not pd.isna(label_df.iloc[ki]['patient_type']):
        now_file = label_df.iloc[ki]['filename'].replace(" ", "")
        now_region = label_df.iloc[ki]['region'].replace(" ", "")
        now_patient_type = label_df.iloc[ki]['patient_type'].replace(" ", "")
        if now_patient_type == 'CD':
            now_class = 1
        else:
            now_class = 0

        now_score = int(label_df.iloc[ki]['score'])
        now_train = 1

        row = len(list_df)
        list_df.loc[row] = [now_file, now_region, now_patient_type, now_score, now_class, now_train]

list_df.to_csv(csv_dir, index = False)