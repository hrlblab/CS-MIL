from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
#from scipy.interpolate import int
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import cv2

attention_df = pd.read_csv('results_stage1_clustering_dataset1/attention_score_dataset1_region.csv')

region_image_folder = '/Data3/GCA_Demo/4864patch_dataset1'
region_folder = '/Data3/GCA_Demo/4864patch_dataset1_patch'
attention_folder = '/Data3/GCA_Demo/4864patch_dataset1_attention'

if not os.path.exists(attention_folder):
    os.makedirs(attention_folder)


patch_size = 256

get_bagscore_flag = 0

if get_bagscore_flag:
    bag_score_df = pd.DataFrame(columns=['bag_num', 'label', 'pred'])
    
    for bi in range(int(len(attention_df)/8)):
        print('%d/%d' % (bi, int(len(attention_df)/8)))
        
        label = 0
        pred = attention_df['score'][bi * 8]
        now_bag = attention_df['bag_num'][bi * 8]
    
        now_root_list = attention_df['root'][bi * 8].split('/')
        region_name = now_root_list[4]
        now_classmark = glob.glob(os.path.join(attention_folder, "%s*classmark.png" % (region_name)))[0]
        img = plt.imread(now_classmark)[:,:,:3]
    
    
        for ki in range(8):
            idx = bi * 8 + ki
            now_root_list = attention_df['root'][idx].split('/')
            region_name = now_root_list[4]
            now_x = int(now_root_list[5].split('_')[1])
            now_y = int(now_root_list[5].split('_')[2])
    
            patch_red = img[now_x - 384:now_x + patch_size - 384, now_y - 384:now_y + patch_size - 384, 0].mean()
            patch_green = img[now_x - 384:now_x + patch_size - 384, now_y - 384:now_y + patch_size - 384, 1].mean()
    
            if patch_red > 0.9 and patch_green < 0.8:
                label = 1
                break
    
        bag_score_df.loc[bi] = [now_bag, label, pred]
    
    bag_score_df.to_csv('results_stage1_clustering_dataset1/bag_score.csv', index = False)

else:
    bag_score_df = pd.read_csv('results_stage1_clustering_dataset1/bag_score.csv')
y_test = np.array(bag_score_df['label'].tolist())
preds = np.array(bag_score_df['pred'].tolist())


# cnt = 0
# for ki in range(len(y_test)):
# 	if (preds[ki] > 0.5 and y_test[ki] == 1) or (preds[ki] < 0.5 and y_test[ki] == 0):
# 		cnt += 1
#
# print(cnt/len(y_test))

acc = ((preds > 0.5) * (y_test == 1) + (preds < 0.5) * (y_test == 0)).mean()


import sklearn.metrics as metrics

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % (roc_auc))
# plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


fpr, tpr, threshold = metrics.roc_curve(y_test, preds, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)


plt.plot(fpr, tpr, linewidth=2, label= '%s, AUC = %0.4f' % ('ours', roc_auc))
plt.legend(loc='lower right')

plt.savefig('ROC-AUC_dataset1.png')
plt.clf()


import matplotlib.pyplot as plt

plt.title('Precision Recall Curve')
# plt.legend(loc='lower right')
plt.plot([0, 1], [0.5, 0.5], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')

precision, recall, threshold = metrics.precision_recall_curve(y_test, preds, pos_label=1)
ap = metrics.average_precision_score(y_test, preds)

plt.plot(recall, precision, linewidth=2,
         label='%s, AP = %0.4f' % ('ours', ap))
plt.legend(loc='lower right')
plt.savefig('PR-AP_dataset1.png')
plt.clf()

print(ap)
print(metrics.accuracy_score(y_test, preds > 0.5))
print(metrics.f1_score(y_test, preds > 0.5))