from __future__ import print_function

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

from DeepAttnMISL_model_CSMIL import DeepAttnMIL_Surv


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

        test_loader = data_utils.DataLoader(
            MILBags(data_csv_root='data_split_nonoverlap_extend/data_list_%d.csv' % (now_split),
                    feature_csv_root='/Data2/GCA/simsiam/feature_cluster_multi-resolution_csv_local_severe_extend_only256',
                    cluster_num=8,
                    size=(256, 256),
                    mean_bag_length=3,
                    cluster_list=[0,1,2,3,4,5,6,7],
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

        output_folder_median = 'results_stage1_clustering_testing_median'
        output_folder_max = 'results_stage1_clustering_testing_max'
        output_folder_mean = 'results_stage1_clustering_testing_mean'

        if not os.path.exists(output_folder_median):
            os.makedirs(output_folder_median)

        if not os.path.exists(output_folder_max):
            os.makedirs(output_folder_max)

        if not os.path.exists(output_folder_mean):
            os.makedirs(output_folder_mean)

        test_acc = 0.
        df_prediction_mean = pd.DataFrame(columns = ['case_name', 'label', 'prob', 'acc'])
        df_prediction_median = pd.DataFrame(columns = ['case_name', 'label', 'prob', 'acc'])
        df_prediction_max = pd.DataFrame(columns = ['case_name', 'label', 'prob', 'acc'])

        softmax_function = nn.Softmax(dim = 1)
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

                'median'
                probs_median = Y_pred_feature.median()

                lbl_pred = (probs_median > threshold).type(torch.float64)
                error = torch.abs(bag_label - lbl_pred)

                acc = 1 - error
                test_acc += acc

                row = len(df_prediction_median)
                df_prediction_median.loc[row] = [now_case, bag_label.item(), probs_median.item(), acc.item()]

                'mean'
                probs_mean = Y_pred_feature.mean()

                lbl_pred = (probs_mean > threshold).type(torch.float64)
                error = torch.abs(bag_label - lbl_pred)

                acc = 1 - error
                test_acc += acc

                row = len(df_prediction_mean)
                df_prediction_mean.loc[row] = [now_case, bag_label.item(), probs_mean.item(), acc.item()]

                'max'
                probs_max = Y_pred_feature.max()

                lbl_pred = (probs_max > threshold).type(torch.float64)
                error = torch.abs(bag_label - lbl_pred)

                acc = 1 - error
                test_acc += acc

                row = len(df_prediction_max)
                df_prediction_max.loc[row] = [now_case, bag_label.item(), probs_max.item(), acc.item()]


            test_acc /= (len(test_loader) * 3)

        print('overall acc: %s' %(str(test_acc.item())))

        df_prediction_mean.to_csv(os.path.join(output_folder_mean,'case_results_%d.csv' % (now_split)), index = False)
        df_prediction_median.to_csv(os.path.join(output_folder_median, 'case_results_%d.csv' % (now_split)), index=False)
        df_prediction_max.to_csv(os.path.join(output_folder_max, 'case_results_%d.csv' % (now_split)), index=False)
