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

def zero_patch_interp(now_score_map, zero_patch_x, zero_patch_y, downrate):
    patch_size = int(1024 / downrate)
    stride = patch_size
    center_bound = 16
    str_x = int((now_score_map.shape[0] - patch_size - start_x) / stride)
    str_y = int((now_score_map.shape[1] - patch_size - start_y) / stride)


    for ii in range(len(zero_patch_x)):
        x_start = zero_patch_x[ii]
        x_end = x_start + patch_size
        y_start = zero_patch_y[ii]
        y_end = y_start + patch_size
    # for xi in range(str_x):
    #     for yi in range(str_y):
    #         if xi != str_x - 1:
    #             x_start = xi * stride + start_x
    #             x_end = xi * stride + patch_size + start_x
    #         else:
    #             continue
    #             #x_start = now_attention_map.shape[0] - patch_size
    #             #x_end = now_attention_map.shape[0]
    #
    #         if yi != str_y - 1:
    #             y_start = yi * stride + start_y
    #             y_end = yi * stride + patch_size + start_y
    #         else:
    #             continue
    #             #y_start = now_attention_map.shape[1] - patch_size
    #             #y_end = now_attention_map.shape[1]

        # print(x_start, y_start)
        # now_channel = now_score_map[x_start:x_end, y_start:y_end]
        # # if now_channel[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
        # #     continue

        mean_channel = 0
        cnt_channel = 0

        neighbour_channel1 = now_score_map[x_start - patch_size: x_end - patch_size, y_start:y_end]
        if neighbour_channel1[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes1')
            mean_channel += neighbour_channel1.mean()
            cnt_channel += 1


        neighbour_channel2 = now_score_map[x_start + patch_size: x_end + patch_size, y_start:y_end]
        if neighbour_channel2[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes2')
            mean_channel += neighbour_channel2.mean()
            cnt_channel += 1


        neighbour_channel3 = now_score_map[x_start: x_end, y_start - patch_size:y_end - patch_size]
        if neighbour_channel3[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes3')
            mean_channel += neighbour_channel3.mean()
            cnt_channel += 1


        neighbour_channel4 = now_score_map[x_start: x_end, y_start + patch_size:y_end + patch_size]
        if neighbour_channel4[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes4')
            mean_channel += neighbour_channel4.mean()
            cnt_channel += 1


        neighbour_channel5 = now_score_map[x_start - patch_size: x_end - patch_size,
                             y_start - patch_size:y_end - patch_size]
        if neighbour_channel5[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes5')
            mean_channel += neighbour_channel5.mean()
            cnt_channel += 1


        neighbour_channel6 = now_score_map[x_start + patch_size: x_end + patch_size,
                             y_start + patch_size:y_end + patch_size]
        if neighbour_channel6[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes6')
            mean_channel += neighbour_channel6.mean()
            cnt_channel += 1


        neighbour_channel7 = now_score_map[x_start - patch_size: x_end - patch_size,
                             y_start + patch_size:y_end + patch_size]
        if neighbour_channel7[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes7')
            mean_channel += neighbour_channel7.mean()
            cnt_channel += 1

        neighbour_channel8 = now_score_map[x_start + patch_size: x_end + patch_size,
                             y_start - patch_size:y_end - patch_size]
        if neighbour_channel8[center_bound:-center_bound, center_bound:-center_bound].sum() != 0:
            print('yes8')
            mean_channel += neighbour_channel8.mean()
            cnt_channel += 1

        if cnt_channel >= 2:
            # print(mean_channel.max() - mean_channel.min())
            mean_value = mean_channel / cnt_channel

            now_score_map[x_start:x_end , y_start:y_end] = mean_value

    return now_score_map



def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def getContour(img):

    img = (resize(img, (1024, 1024)) * 255).astype(np.uint8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space

    mthresh = 5
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring


    sthresh = 20
    sthresh_up = 255

    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    _, contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

    contours_tissue = foreground_contours
    holes_tissue = hole_contours

    tissue_mask = get_seg_mask(region_size = img.shape, scale = 0, contours_tissue = contours_tissue, holes_tissue = holes_tissue, use_holes=True, offset=(0, 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res2 = resize(cv2.dilate(tissue_mask,kernel, iterations = 10), (4864,4864))

    return res2


def get_seg_mask(region_size, scale, contours_tissue, holes_tissue, use_holes=False, offset=(0, 0)):
    # print('\ncomputing foreground tissue mask')
    tissue_mask = np.full(region_size,0).astype(np.uint8)
    offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
    contours_holes = holes_tissue
    contours_tissue, contours_holes = zip(
        *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
    for idx in range(len(contours_tissue)):
        cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1,1,1), offset=offset,
                         thickness=-1)

        if use_holes:
            cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0,0,0),
                             offset=offset, thickness=-1)

    return tissue_mask.astype(np.float32)



attention_df = pd.read_csv('/Data3/GCA_Demo/Image_stack#21/results_stage1_clustering_dataset1/attention_score_dataset1_region.csv')


region_image_folder = '/Data3/GCA_Demo/4864patch_dataset1'
region_folder = '/Data3/GCA_Demo/4864patch_dataset1_patch'
attention_folder = '/Data3/GCA_Demo/4864patch_dataset1_attention'

if not os.path.exists(attention_folder):
    os.makedirs(attention_folder)

classmarks = glob.glob(os.path.join(attention_folder, "*classmark.png"))

'get bbox'
import seaborn as sns
from scipy.stats import wilcoxon

x_20_class0 = []
x_10_class0 = []
x_5_class0 = []

x_20_class1 = []
x_10_class1 = []
x_5_class1 = []

patch_size = 256

for ci in range(len(classmarks)):
    img = plt.imread(classmarks[ci])

    slide_name = classmarks[ci].split('/')[-1].replace('_classmark.png','')

    now_wsi_list = slide_name.split('_')
    now_wsi_name = '%s_%s_%s_%s_%s_%s_%s' % (now_wsi_list[0],now_wsi_list[1],now_wsi_list[2],now_wsi_list[3],now_wsi_list[4],now_wsi_list[5],now_wsi_list[6])
    df_score = attention_df.loc[attention_df['root'].str.contains(now_wsi_name, case=False)]
    region_name = '.png_%s_%s' % (now_wsi_list[-2],now_wsi_list[-1])
    df_now = df_score.loc[df_score['root'].str.contains(region_name, case=False)]

    for xi in range(0,16):
        for yi in range(0,16):
            now_x = xi * 256 + 384
            now_y = yi * 256 + 384

            now_score_list =  df_now.loc[df_now['root'].str.contains('region_%d_%d_size1024.png' % (now_x, now_y), case=False)]
            if len(now_score_list) > 0:


            # for di in range(len(df_now)):
            #     now_row = df_now.iloc[di]
            #     now_x = int(now_row['root'].split('/')[-1].split('_')[-3])
            #     now_y = int(now_row['root'].split('/')[-1].split('_')[-2])

                patch_red = img[now_x - 384:now_x + patch_size - 384, now_y - 384:now_y + patch_size - 384, 0].mean()
                patch_green = img[now_x - 384:now_x + patch_size - 384, now_y - 384:now_y + patch_size - 384, 1].mean()
                #print(patch_red, patch_green)

                if patch_red > 0.9 and patch_green < 0.8:
                    x_20_class1.append(now_score_list['20X'].mean())
                    x_10_class1.append(now_score_list['10X'].mean())
                    x_5_class1.append(now_score_list['5X'].mean())

                else:
                    x_20_class0.append(now_score_list['20X'].mean())
                    x_10_class0.append(now_score_list['10X'].mean())
                    x_5_class0.append(now_score_list['5X'].mean())


# df_1 = attention_df[attention_df['instance_score'] > 0.20]
#
# df_1 = df_1[df_1['score'] > 0.5]
#
# df_1.head()

# sns.boxplot( x=df['20X'], y=df['20X'])
# plt.show()

fig, ax = plt.subplots()
plt.ylim([0,1])
df = pd.DataFrame(list(zip(x_20_class1, x_10_class1, x_5_class1)), columns = ['20X', '10X', '5X'])

sns.boxplot(data=df[['20X', '10X', '5X']], ax=ax)
plt.rcParams["figure.figsize"] = (3,3)
plt.savefig('No_multiple_instance_only1_dataset1.pdf')

X20_X10 = wilcoxon(x_20_class1, x_10_class1)
X20_X5 = wilcoxon(x_20_class1, x_5_class1)
X10_X5 = wilcoxon(x_10_class1, x_5_class1)

print(X20_X10,X20_X5,X10_X5)

#
#
# df_1 = attention_df#[attention_df['instance_score'] > 0.20]
#
# df_1 = df_1[df_1['score'] < 0.5]
#
# df_1.head()

# sns.boxplot( x=df['20X'], y=df['20X'])
# plt.show()
plt.close()

fig, ax = plt.subplots()
plt.ylim([0,1])
df = pd.DataFrame(list(zip(x_20_class0, x_10_class0, x_5_class0)),columns = ['20X', '10X', '5X'])

sns.boxplot(data=df[['20X', '10X', '5X']], ax=ax)
plt.rcParams["figure.figsize"] = (3,3)

plt.savefig('No_multiple_instance_only0_dataset1.pdf')

X20_X10 = wilcoxon(x_20_class0, x_10_class0)
X20_X5 = wilcoxon(x_20_class0, x_5_class0)
X10_X5 = wilcoxon(x_10_class0, x_5_class0)

print(X20_X10,X20_X5,X10_X5)

#
# df_1 = pd.DataFrame(columns = ['20X', '10X', '5X'])
#
# df_1['20X'] = attention_df['instance_score'] * attention_df['20X']
# df_1['10X'] = attention_df['instance_score'] * attention_df['10X']
# df_1['5X'] = attention_df['instance_score'] * attention_df['5X']
#
# df_1 = attention_df[attention_df['instance_score'] > 0.20]
#
# fig, ax = plt.subplots()
#
#
# df_1 = df_1[df_1['score'] > 0.5]
#
# sns.boxplot(data=df_1[['20X','10X','5X']], ax=ax)
#
# plt.savefig('multiple_instance_all.png')
fig, ax = plt.subplots()
plt.ylim([0,1])
df = pd.DataFrame(list(zip(x_20_class1, x_10_class1, x_5_class1)), columns = ['20X', '10X', '5X'])

sns.boxplot(data=df[['20X', '10X', '5X']], ax=ax)
plt.rcParams["figure.figsize"] = (3,3)
plt.savefig('No_multiple_instance_only1_dataset1.pdf')