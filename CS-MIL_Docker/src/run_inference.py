from __future__ import print_function
'packages for step1 getting patches'
import glob
import timeit
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import os

import openslide
from skimage.transform import resize
import PIL
from PIL import Image
import tifffile
import scipy.ndimage as ndi
#import matplotlib._png as png
from matplotlib.cbook import get_sample_data



'packages for step2 embedding features'

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
import numpy as np
import random


'packages for step3 clustering features'

from sklearn.cluster import KMeans
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt


'packages for step4 get CS-MIL results'

import numpy as np

import argparse

import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from MIL_dataloader_csv_MAg_clustering_pair_stack import MILBags

import copy
import matplotlib.pyplot as plt

from DeepAttnMISL_CS_MIL import DeepAttnMIL_Surv

import glob
import os

from torch import nn

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

def getContour(img, contour_dir, case_filename, down_rate, patch_size, wsi_level, level_down_rate, start):

	shape1 = int(img.level_dimensions[wsi_level][0])
	shape0 = int(img.level_dimensions[wsi_level][1])

	img_otsu = np.zeros((int(shape0 / down_rate), int(shape1 / down_rate))).astype(np.uint8)

	img_png = (255 * np.asarray(img.read_region(start, wsi_level + level_down_rate, img.level_dimensions[wsi_level + level_down_rate]))[:, :, :3]).astype(np.uint8)

	x_stride = int(img_png.shape[0] / (patch_size)) + 1
	y_stride = int(img_png.shape[1] / (patch_size)) + 1

	for xi in range(x_stride):
		for yi in range(y_stride):
			x_ind = xi * patch_size
			y_ind = yi * patch_size

			print(xi)
			print(yi)
			img_down = img_png[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size]

			img_hsv = cv2.cvtColor(img_down, cv2.COLOR_RGB2HSV)  # Convert to HSV space

			mthresh = 59
			img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

			sthresh = 20
			sthresh_up = 255

			shape_x = img_med.shape[0]
			shape_y = img_med.shape[1]

			if shape_x != patch_size or shape_y != patch_size:
				_, img_otsu[x_ind:x_ind + shape_x, y_ind:y_ind + shape_y] = cv2.threshold(img_med, sthresh,sthresh_up,cv2.THRESH_BINARY)
			else:
				_, img_otsu[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size] = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)


	# # Morphological closing
	# if close > 0:
	close = 1
	kernel = np.ones((close, close), np.uint8)
	img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

	# Find and filter contours
	contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
	hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

	filter_params = {'a_t':20, 'a_h': 16, 'max_n_holes':8}
	foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

	contours_tissue = foreground_contours
	holes_tissue = hole_contours

	#tissue_mask[x_ind:x_ind+patch_size,y_ind:y_ind+patch_size,:] = get_seg_mask(region_size = img_down.shape, scale = 0, contours_tissue = contours_tissue, holes_tissue = holes_tissue, use_holes=True, offset=(0, 0))
	tissue_mask = get_seg_mask(region_size = (img_otsu.shape[0], img_otsu.shape[1], 3), scale = 0, contours_tissue = contours_tissue, holes_tissue = holes_tissue, use_holes=True, offset=(0, 0))

	del img
	del img_down
	del img_otsu
	del img_hsv
	del img_med
	output_root = os.path.join(contour_dir, case_filename)

	plt.imsave(output_root, tissue_mask)

	# np.save(image_dir, tissue_mask, allow_pickle=True)

	return tissue_mask[:, :, :1], output_root


def get_seg_mask(region_size, scale, contours_tissue, holes_tissue, use_holes=False, offset=(0, 0)):
	print('\ncomputing foreground tissue mask')
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

	# # tissue_mask = tissue_mask.astype(bool)
	# print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
	return tissue_mask.astype(np.float32)


def scan_nonblack_end(simg, px_start, py_start, px_end, py_end):
	offset_x = 0
	offset_y = 0
	line_x = py_end - py_start
	line_y = px_end - px_start

	val = simg.read_region((px_end + offset_x, py_end), 0, (1, 1))
	arr = np.array(val)[:, :, 0].sum()
	while not arr == 0:
		val = simg.read_region((px_end + offset_x, py_end), 0, (1, line_x))
		arr = np.array(val)[:, :, 0].sum()
		offset_x = offset_x + 1

	val = simg.read_region((px_end, py_end + offset_y), 0, (1, 1))
	arr = np.array(val)[:, :, 0].sum()
	while not arr == 0:
		val = simg.read_region((px_end, py_end + offset_y), 0, (line_y, 1))
		arr = np.array(val)[:, :, 0].sum()
		offset_y = offset_y + 1

	x = px_end + (offset_x - 1)
	y = py_end + (offset_y - 1)
	return x, y

def crop_patches_cpu(img, contour_map, patch_folder, down_rate, wsi_level, start):

	patch_size = 256
	stride_size = 256
	'get mult-scale tiles'

	shape1 = int(img.level_dimensions[wsi_level][0])
	shape0 = int(img.level_dimensions[wsi_level][1])

	stride_x = int(shape0 / stride_size) + 1
	stride_y = int(shape1 / stride_size) + 1

	for xi in range(2,stride_x - 2):
		for yi in range(2, stride_y - 2):

			x_ind = int(xi * stride_size)
			y_ind = int(yi * stride_size)

			'wsi takes ori abs coordinates'
			x_ind_large = x_ind * (2 ** wsi_level)
			y_ind_large = y_ind * (2 ** wsi_level)

			# contour_patch = contour_map[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size, :]
			contour_patch = contour_map[int(x_ind / down_rate) :int((x_ind + patch_size) / down_rate), int(y_ind / down_rate):int( (y_ind + patch_size) / down_rate), :]
			if 1 in contour_patch:

				now_patch_256 = np.asarray(img.read_region((y_ind_large + start[1], x_ind_large + start[0]),wsi_level,(256, 256)))[:,:,:3]
				now_patch_512 = np.asarray(img.read_region((y_ind_large - 256 + start[1], x_ind_large - 256 + start[0]),wsi_level,(512, 512)))[:,:,:3]
				now_patch_1024 = np.asarray(img.read_region((y_ind_large - 768 + start[1], x_ind_large - 768 + start[0]), wsi_level,(1024,1024)))[:,:,:3]

				# now_patch_256 = np.asarray(img.read_region((y_ind + start[0], x_ind + start[1]),wsi_level,(256, 256)))[:,:,:3]
				# now_patch_512 = np.asarray(img.read_region((y_ind - 128 + start[0], x_ind - 128 + start[1]),wsi_level,(512, 512)))[:,:,:3]
				# now_patch_1024 = np.asarray(img.read_region((y_ind - 384 + start[0], x_ind - 384 + start[1]), wsi_level,(1024,1024)))[:,:,:3]


				patch_dir_256 = os.path.join(patch_folder, '%d_%d_size256.png' % (x_ind, y_ind))
				patch_dir_512 = os.path.join(patch_folder, '%d_%d_size512.png' % (x_ind, y_ind))
				patch_dir_1024 = os.path.join(patch_folder, '%d_%d_size1024.png' % (x_ind, y_ind))

				plt.imsave(patch_dir_256, now_patch_256)
				plt.imsave(patch_dir_512, now_patch_512)
				plt.imsave(patch_dir_1024, now_patch_1024)


def getFeatures_embedding(args, patch_folder, feature_folder):
	tensor_transform = T.Compose([
		T.Resize([256,]),
		# T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	resolution_list = [256,512,1024]
	for rrri in range(len(resolution_list)):

		resolution = resolution_list[rrri]

		if resolution == 1024:
			args.eval_from = '/myhome/checkpoint/simsiam-GCA-b6128s256_ori1024patch_0414131202.pth'
			#args.eval_from = './checkpoint/simsiam-GCA-b6128s256_ori1024patch_0414131202.pth'
		elif resolution == 512:
			args.eval_from = '/myhome/checkpoint/simsiam-GCA-b6128s256_ori512patch_0418160825.pth'
			# args.eval_from = './checkpoint/simsiam-GCA-b6128s256_ori512patch_0418160825.pth'
		else:
			args.eval_from = '/myhome/checkpoint/simsiam-GCA-b6128s256_ori256patch_0422134301.pth'
			# args.eval_from = './checkpoint/simsiam-GCA-b6128s256_ori256patch_0422134301.pth'


		model = get_backbone(args.model.backbone)

		assert args.eval_from is not None
		save_dict = torch.load(args.eval_from, map_location='cpu')
		msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)

		# print(msg)
		model = model.to(args.device)
		model = torch.nn.DataParallel(model)

		model.eval()

		# for row in range(len(label_file)):
		#     now_case = label_file.iloc[row]['filename'].replace('.svs','')
		#     now_images = glob.glob(os.path.join(data_dir, now_case, '*size%d.png' % (resolution)))
		#     images.extend((now_images))

		images = glob.glob(os.path.join(patch_folder, '*size%d.png' % (resolution)))
		random.shuffle(images)
		patch_size = resolution
		batch = 32 #int(args.eval.batch_size)
		bag_num = int(len(images) / batch) + 1

		if not os.path.exists(feature_folder):
			os.makedirs(feature_folder)

		with torch.no_grad():
			for ri in range(bag_num):
				if ri != bag_num - 1:
					now_images = images[ri * batch : (ri + 1) * batch]
				else:
					now_images = images[ri * batch:]

				tensor = np.zeros((len(now_images), patch_size , patch_size, 3))
				for ni in range(len(now_images)):
					image_dir = now_images[ni]
					tensor[ni] = plt.imread(image_dir)[:,:,:3]

				# tensor = tensor.transpose([0,3,1,2])
				tensor = torch.from_numpy(tensor).permute([0,3,1,2])
				inputs = tensor_transform(tensor)
				features = model(inputs.to(args.device).float())

				for fi in range(len(features)):
					now_name = os.path.basename(now_images[fi])
					wsi_name = now_name.split('_')[0]
					now_feature = features[fi].detach().cpu().numpy()

					if not os.path.exists(feature_folder):
						os.makedirs(feature_folder)

					save_dir = os.path.join(feature_folder, now_name.replace('.png', '.npy'))
					np.save(save_dir, now_feature)


def getFeatures_clustering(feature_folder, cluster_folder, wsi_name):
	resolution_list = [256]

	for rl in range(len(resolution_list)):
		resolution = resolution_list[rl]

		print('now is clustering : resolution %d' % (resolution))

		feature_files = glob.glob(os.path.join(feature_folder,"*size%d.npy" % (resolution)))

		features = np.zeros((len(feature_files), 2048))

		for ii in range(len(feature_files)):
			features[ii] = np.load(feature_files[ii])

		local_features = features
		local_features_file = feature_files

		kmeans = KMeans(n_clusters=8, random_state=0).fit(local_features)
		df = pd.DataFrame(columns=['wsi_name', 'root', 'label', 'cluster'])
		for ii in range(len(local_features)):
			now_root = local_features_file[ii]
			now_wsi = wsi_name
			now_label = 0

			now_cluster = kmeans.labels_[ii]
			row = len(df)
			df.loc[row] = [now_wsi, now_root, now_label, now_cluster]

		save_root = os.path.join(cluster_folder, '%s_cluster.csv' % (now_wsi))
		df.to_csv(save_root, index=False)


def Testing(data_list_root, cluster_folder, result_folder):

	data_split = 10
	final_testing_acc = []
	for di in range(0, data_split):
		now_split = di

		# Training settings
		parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
		parser.add_argument('--epochs', type=int, default=100, metavar='N',
							help='number of epochs to train (default: 20)')
		parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
							help='learning rate (default: 0.0005)')
		parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
							help='weight decay')
		parser.add_argument('--target_number', type=int, default=9, metavar='T',
							help='bags have a positive labels if they contain at least one 9')
		parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
							help='average bag length')
		parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
							help='variance of bag length')
		parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
							help='number of bags in training set')
		parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
							help='number of bags in test set')
		parser.add_argument('--seed', type=int, default=1, metavar='S',
							help='random seed (default: 1)')
		parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
		parser.add_argument('--model', type=str, default='attention',
							help='Choose b/w attention and gated_attention')

		args = parser.parse_args()
		args.cuda = not args.no_cuda and torch.cuda.is_available()

		torch.manual_seed(args.seed)
		if args.cuda:
			torch.cuda.manual_seed(args.seed)
			# print('\nGPU is ON!')

		# print('Load Train and Test Set')
		loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

		test_loader = data_utils.DataLoader(
			MILBags(data_csv_root=data_list_root,
					feature_csv_root=cluster_folder,
					cluster_num=8,
					size=(256, 256),
					mean_bag_length=3,
					cluster_list=[0, 1, 2, 3, 4, 5, 6, 7],
					var_bag_length=1,
					seed=1,
					train=2),
			batch_size=1,
			shuffle=False)

		# print('Init Model')
		cluster_num = 8
		model1 = DeepAttnMIL_Surv(cluster_num=cluster_num)

		MODEL_PATH = glob.glob(os.path.join('/myhome/MIL_models', '%d_model*.pth' % (now_split)))[0]
		#MODEL_PATH = glob.glob(os.path.join('./MIL_models', '%d_model*.pth' % (now_split)))[0]

		save_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
		# model = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
		#                             strict=True)

		model1.load_state_dict(save_dict)
		model1.eval()

		threshold = 0.5

		if args.cuda:
			model1.cuda()
		# model2.cuda()

		output_folder = '/myhome/results_clustering_testing'
		#output_folder = './results_clustering_testing'

		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		test_acc = 0.
		df_prediction_mean = pd.DataFrame(columns=['case_name', 'label', 'prob', 'acc'])

		# softmax_function = nn.Softmax(dim=1)
		with torch.no_grad():
			for batch_idx, (data, label, mask, case_name, skip) in enumerate(test_loader):
				if skip:
					continue
				print('%d/%d' % (batch_idx + 1, len(test_loader)))
				bag_label = label[0][0].cuda()
				# graph = [now_data[0].unsqueeze(2).unsqueeze(2) for now_data in data]
				now_case = case_name[0]
				Y_pred_feature = torch.zeros((len(data))).cuda()

				for ki in range(len(data)):
					now_data = data[ki]
					graph = [now_data1[0] for now_data1 in now_data]
					Y_pred_feature[ki] = model1(graph, mask[ki].cuda())

				# hist = torch.histc(Y_pred_feature, bins=10, min=0, max=1).detach().cuda()
				# hist = hist / len(data)
				# lbl_probs = model2(hist.unsqueeze(0))

				'mean'
				probs_mean = Y_pred_feature.mean()

				lbl_pred = (probs_mean > threshold).type(torch.float64)
				error = torch.abs(bag_label - lbl_pred)

				acc = 1 - error
				test_acc += acc

				row = len(df_prediction_mean)
				df_prediction_mean.loc[row] = [now_case, bag_label.item(), probs_mean.item(), acc.item()]

			test_acc /= (len(test_loader) * 3)

		print('case: %s, num: %s,  pred: %s' % (now_case, str(di), str(df_prediction_mean['prob'].mean())))

		df_prediction_mean.to_csv(os.path.join(result_folder, 'case_results_%d.csv' % (now_split)), index=False)


	df_merge = pd.DataFrame(columns=['case_name', 'label', 'prob', 'acc'])
	for di in range(data_split):
		now_df = pd.read_csv(os.path.join(result_folder, 'case_results_%d.csv' % (di)))

		for pi in range(len(now_df)):
			now_row = now_df.loc[pi]
			now_case = now_row['case_name']

			row = len(df_merge)
			df_merge.loc[row] = now_row

	row = len(df_merge)
	print('overall prediction: %s' % (df_merge['prob'].mean()))
	df_merge.loc[row] = ['overall', 0, df_merge['prob'].mean(), 0]
	df_merge.to_csv(os.path.join(result_folder, 'case_results_merge.csv'), index=False)


if __name__ == "__main__":

	input_dir = '/input/'
	output_dir = '/output/'
	wd = '/myhome/wd'

	step1 = 1
	step1_contour = 1
	step1_png = 0
	step1_tile = 1
	step2 = 1
	step3 = 1
	step4 = 1

	# data_dir = '/Data2/CS-MIL_Docker/src/input'
	# output_dir = '/Data2/CS-MIL_Docker/src/output'
	# contour_dir = '/Data2/CS-MIL_Docker/src/output/tissue_seg'
	# slide_dir = '/Data2/CS-MIL_Docker/src/output/slides'
	# patch_dir = '/Data2/CS-MIL_Docker/src/output/patches'
	# feature_dir = '/Data2/CS-MIL_Docker/src/output/features'
	# cluster_dir = '/Data2/CS-MIL_Docker/src/output/clusters'
	# result_dir = '/Data2/CS-MIL_Docker/src/output/results'
	
	
	data_dir = input_dir
	output_dir = output_dir#os.path.join(output_dir,'output')
	contour_dir = os.path.join(output_dir,'tissue_seg')
	slide_dir = os.path.join(output_dir,'slides')
	patch_dir = os.path.join(output_dir,'patches')
	feature_dir = os.path.join(output_dir,'features')
	cluster_dir = os.path.join(output_dir,'clusters')
	result_dir = os.path.join(output_dir,'results')

	sections = glob.glob(os.path.join(data_dir, '*'))
	sections.sort()

	print(result_dir)

	for si in range(0,len(sections)):

		name = os.path.basename(sections[si])
		ext = os.path.splitext(name)[1]
		print(name)

		'check whether they have results already'
		if os.path.exists(os.path.join(result_dir,name.replace(ext, ''),'case_results_merge.csv')):
			print('This case is done')
			continue

		# if name == 'S18-2001 - 2023-07-31 21.40.05.ndpi':
		# 	print('pass this case')
		# 	continue

		filename_20X = os.path.join(slide_dir, name.replace(ext, '.png'))
		if not os.path.exists(os.path.dirname(filename_20X)):
			os.makedirs(os.path.dirname(filename_20X))

		if not os.path.exists(contour_dir):
			os.makedirs(contour_dir)

		patch_folder = os.path.join(patch_dir, name.replace(ext, ''))
		if not os.path.exists(patch_folder):
			os.makedirs(patch_folder)

		'''''''''''''''''''''''''''
		'step 1 get patch from WSI'
		'''''''''''''''''''''''''''
		if step1:
			start1 = timeit.default_timer()
			'get the image dimension'
			img = openslide.open_slide(sections[si])
			if 'ndpi' in sections[si]:
				mag = int(img.properties['hamamatsu.SourceLens'])
			else:
				mag = int(img.properties['aperio.AppMag'])

			x, y = scan_nonblack_end(img, 0, 0, img.dimensions[0], img.dimensions[1])
			start = img.dimensions - np.array((x, y))
			print(start)

			if mag == 40:
				wsi_level = 1
			elif mag == 20:
				wsi_level = 0

			if step1_contour:
				'get foreground contour'
				if 'ndpi' in sections[si]:
					down_rate = 8
					level_down_rate = 3
					patch_size = 4096
				else:
					down_rate = 4
					level_down_rate = 1
					patch_size = 4096

				contour_map, now_contour_dir = getContour(img, contour_dir, name.replace(ext, '.png'), down_rate, patch_size, wsi_level, level_down_rate, start)

			if step1_png:
				'get 20X img to PNG for scn/svs'
				# if mag == 20:
				# 	img_20X = np.asarray(img.read_region(start, 0, img.dimensions))[:,:,:3]
				# elif mag == 40:
				# 	img_40X = np.asarray(img.read_region(start, 0, img.dimensions))[:,:,:3]
				# 	img_20X = resize(img_40X, (int(img_40X.shape[0] / 2), int(img_40X.shape[1] / 2)))

				'get 20X img to PNG for ndpi'
				if mag == 40:
					img_20X = np.asarray(img.read_region(start, wsi_level, img.level_dimensions[wsi_level]))[:, :, :3]
				elif mag == 20:
					img_20X = np.asarray(img.read_region(start, wsi_level, img.level_dimensions[wsi_level]))[:, :, :3]

				# del img

				if img_20X.max() == 1.:
					img_20X = (img_20X / img_20X.max() * 255).astype(np.uint8)

				print(img_20X.shape)
				try:
					plt.imsave(filename_20X, img_20X)
					print('USE plt')
				except:
					cv2.imwrite(filename_20X, img_20X)
					print('USE cv2')
				del img_20X
			else:
				print('dont need to get png')

			if step1_tile:
				contour_map[contour_map > 0.5] = 1
				contour_map[contour_map < 0.5] = 0

				contour_map = contour_map.astype(np.uint8)

				print(contour_map.shape)

				crop_patches_cpu(img, contour_map, patch_folder, down_rate, wsi_level, start)

				del contour_map
				#del image

			end1 = timeit.default_timer()
			print('getting patch duration:', end1 - start1, 'seconds')

		'''''''''''''''''''''''''''''
		'step 2 embedding the patches'
		'''''''''''''''''''''''''''''
		feature_folder = os.path.join(feature_dir, name.replace(ext, ''))
		if not os.path.exists(feature_folder):
			os.makedirs(feature_folder)


		if step2:
			start2= timeit.default_timer()
			args = get_args()
			getFeatures_embedding(args, patch_folder, feature_folder)

			end2 = timeit.default_timer()
			print('embedding tiles:', end2 - start2, 'seconds')


		''''''''''''''''''''''''''''''''
		'step 3 clustering the features'
		''''''''''''''''''''''''''''''''
		cluster_folder = os.path.join(cluster_dir, name.replace(ext, ''))
		if not os.path.exists(cluster_folder):
			os.makedirs(cluster_folder)


		if step3:
			start3 = timeit.default_timer()
			getFeatures_clustering(feature_folder, cluster_folder, name.replace(ext, ''))

			end3 = timeit.default_timer()
			print('clustering tiles:', end3 - start3, 'seconds')


		'''''''''''''''''''''''''''
		'step 4 Get CS-MIL results'
		'''''''''''''''''''''''''''
		result_folder = os.path.join(result_dir, name.replace(ext, ''))
		if not os.path.exists(result_folder):
			os.makedirs(result_folder)


		if step4:
			start4 = timeit.default_timer()

			data_list = pd.DataFrame(columns=['filename', 'region', 'patient_type', 'score', 'class', 'train'])
			data_list.loc[0] = [name.replace(ext, ''), 'colon', 'n/a', 0, 0, 2]
			data_list_root = os.path.join(result_folder, 'data_list.csv')
			data_list.to_csv(data_list_root, index=False)

			Testing(data_list_root, cluster_folder, result_folder)
			end4 = timeit.default_timer()
			print('testing tiles:', end4 - start4, 'seconds')
