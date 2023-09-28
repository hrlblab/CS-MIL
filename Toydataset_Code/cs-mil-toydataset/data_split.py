import numpy as np
import pandas as pd
import random

pd.options.mode.chained_assignment = None
split_num = 5
original_file = pd.read_csv('/Data2/GCA/AttentionDeepMIL/data_split/data_list.csv')
class1 = original_file[original_file['class'] == 1]['filename'].tolist()
class0 = original_file[original_file['class'] == 0]['filename'].tolist()

test_size = 4

for si in range(split_num):
    random.shuffle(class0)
    random.shuffle(class1)
    # ind_0 = np.random.randint(0, len(class0), test_size)
    # ind_1 = np.random.randint(0, len(class1), test_size)

    testing_0 = class0[:11]
    testing_1 = class1[:11]

    validation_0 = class0[11:14]
    validation_1 = class1[11:14]

    training_0 = class0[14:]
    training_1 = class1[14:]

    for oi in range(len(original_file)):
        if (original_file.iloc[oi]['filename'] in testing_0) or (original_file.iloc[oi]['filename'] in testing_1):
            original_file.iloc[oi, original_file.columns.get_loc('train')] = 2
        elif (original_file.iloc[oi]['filename'] in validation_0) or (original_file.iloc[oi]['filename'] in validation_1):
            original_file.iloc[oi, original_file.columns.get_loc('train')] = 1
        else:
            original_file.iloc[oi, original_file.columns.get_loc('train')] = 0
    original_file.to_csv('/Data2/GCA/AttentionDeepMIL/data_split/data_list_%d.csv' % (si), index = False)






