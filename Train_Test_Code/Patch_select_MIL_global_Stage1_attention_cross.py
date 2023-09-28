from __future__ import print_function

import numpy as np

import argparse

import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from Patch_select_MIL_dataloader_csv_MAg_clustering_check_stack import MILBags

import copy
import matplotlib.pyplot as plt

from DeepAttnMISL_model_no21_attentionscore import DeepAttnMIL_Surv
from Classifier_model_MAg import Classifier

import glob
import os

from torch import nn

if __name__ == "__main__":

    data_split = 10
    final_testing_acc = []
    for ki in range(0,data_split):
        now_split = ki

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
        parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            print('\nGPU is ON!')

        print('Load Train and Test Set')
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        test_loader = data_utils.DataLoader(MILBags(data_csv_root='/Data2/GCA/Severe_AttentionDeepMIL_localclustering_multi-resolution_pair_stack#21/data_split_nonoverlap_extend/data_list_%d.csv' % (now_split),
                                                    feature_csv_root='/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_severe_extend_only256',
                                                    cluster_num=8,
                                                    cluster_start=0,
                                                    size=(256, 256),
                                                    cluster_list=[0,1,2,3,4,5,6,7],
                                                    mean_bag_length=3,
                                                    var_bag_length=1,
                                                    seed=1,
                                                    train=2),
                                            batch_size=1,
                                            shuffle=False)

        print('Init Model')
        cluster_num = 8
        model1 = DeepAttnMIL_Surv(cluster_num=cluster_num)

        MODEL_PATH = glob.glob(os.path.join('results_stage1_clustering', '%d_model*.pth' % (now_split)))[0]

        save_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
        # model = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
        #                             strict=True)

        model1.load_state_dict(save_dict)
        model1.eval()

        threshold = 0.5

        if args.cuda:
            model1.cuda()
            # model2.cuda()

        output_folder = 'Patches_Attention'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        test_acc = 0.
        # df_prediction = pd.DataFrame(columns = ['case_name', 'label', 'prob', 'acc'])

        softmax_function = nn.Softmax(dim = 1)
        with torch.no_grad():
            for batch_idx, (data, label, mask, case_name, feature_root, skip) in enumerate(test_loader):
                if skip:
                    continue
                print('%d/%d' % (batch_idx + 1, len(test_loader)))
                bag_label = label[0][0].cuda()
                # graph = [now_data[0].unsqueeze(2).unsqueeze(2) for now_data in data]
                now_case = case_name[0]
                #Y_pred_feature = torch.zeros((len(data))).cuda()

                df_features = pd.DataFrame(columns=['feature_root', 'attention_256', 'attention_512', 'attention_1024', 'attention_instance'])

                for ki in range(len(data)):
                    now_data = data[ki]
                    graph = [now_data1[0] for now_data1 in now_data]
                    _, resolution_attention_256, resolution_attention_512, resolution_attention_1024, A = model1(graph, mask[ki].cuda())

                    cntt = 0
                    for cls in range(len(feature_root[ki])):
                        now_feature_root_cluster = feature_root[ki][cls]
                        now_resolution_attention_256_cluster = resolution_attention_256[cls]
                        now_resolution_attention_512_cluster = resolution_attention_512[cls]
                        now_resolution_attention_1024_cluster = resolution_attention_1024[cls]

                        for fi in range(len(now_feature_root_cluster)):
                            now_feature = now_feature_root_cluster[fi]

                            now_resolution_attention_256 = now_resolution_attention_256_cluster[fi]
                            now_resolution_attention_512 = now_resolution_attention_512_cluster[fi]
                            now_resolution_attention_1024 = now_resolution_attention_1024_cluster[fi]

                            Softmax = nn.Softmax(dim=0)

                            resolution_attention_cat = torch.cat([now_resolution_attention_256, now_resolution_attention_512, now_resolution_attention_1024], 0)
                            resolution_attention_softmax = Softmax(resolution_attention_cat)
                            now_attention_256 = resolution_attention_softmax[0].item()
                            now_attention_512 = resolution_attention_softmax[1].item()
                            now_attention_1024 = resolution_attention_softmax[2].item()

                            now_instance_attention = A[:,cntt].item()

                            if now_feature != 'aaa':
                                # now_score = Y_pred_feature[ki].item()
                                row = len(df_features)
                                df_features.loc[row] = [now_feature, now_attention_256, now_attention_512, now_attention_1024, now_instance_attention]

                            cntt += 1

                df_merge = pd.DataFrame(columns=['feature_root', 'attention_256', 'attention_512', 'attention_1024', 'attention_instance'])

                for ri in range(len(df_features)):
                    now_feature = df_features.iloc[ri]

                    if len(df_merge[df_merge['feature_root'] == now_feature['feature_root']]) == 0:
                        temp = df_features[df_features['feature_root'] == now_feature['feature_root']]
                        mean_nattention_256 = np.mean(temp['attention_256'].tolist())
                        mean_nattention_512 = np.mean(temp['attention_512'].tolist())
                        mean_nattention_1024 = np.mean(temp['attention_1024'].tolist())
                        mean_nattention_instance = np.mean(temp['attention_instance'].tolist())
                        row = len(df_merge)
                        df_merge.loc[row] = [now_feature['feature_root'], mean_nattention_256, mean_nattention_512, mean_nattention_1024, mean_nattention_instance]

                save_folder = os.path.join(output_folder)
                # df_merge.sort_values(by='cnt', ascending=False)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_root = os.path.join(save_folder, '%s_model_%d.csv' % (now_case, now_split))
                df_merge.to_csv(save_root, index = False)