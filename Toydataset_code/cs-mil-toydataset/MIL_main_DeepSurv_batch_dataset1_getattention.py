from __future__ import print_function

import argparse

import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from MIL_dataloader_csv_data1 import MILBags
import os

import copy
import matplotlib.pyplot as plt

from DeepAttnMISL_model_no21_withResnet import DeepAttnMIL_Surv
import glob

def calculate_loss(Y_prob, Y):
    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

    return neg_log_likelihood


def test(epoch, model, output_folder, batch_size = 4, mini_batch = 4):

    'do validation'
    model.eval()
    val_loss = 0.
    val_error = 0.
    
    df = pd.DataFrame(columns=['bag_num','root','instance_score','class','score'])
    
    with torch.no_grad():
        for batch_idx, (data, label, mask, root) in enumerate(validation_loader):
            print('%d / %d' % (batch_idx, len(validation_loader)))
            bag_label = label[0].cuda()

            # calculate loss and metrics
            lbl_probs, instance_attention, scale_attention = model(data.cuda(), mask.cuda())

            loss = calculate_loss(lbl_probs, bag_label)
            val_loss += loss.data[0]

            lbl_pred = (lbl_probs > 0.5).type(torch.float64)
            error = torch.abs(bag_label - lbl_pred[0])

            val_error += error
            
            softmax = torch.nn.Softmax()
            
            for ki in range(len(root)):
                now_bag_num = batch_idx
                now_root = root[ki]
                now_instance_score = instance_attention[0,ki].item()
                
                row = len(df)
                df.loc[row] = [now_bag_num, now_root, now_instance_score, bag_label.item(), lbl_probs[0].item()]

        df.to_csv(os.path.join(output_folder,'attention_score_dataset1_region.csv'), index = False)
        val_error /= len(validation_loader)
        val_loss /= len(validation_loader)

        # print('==========================================================================================================')
        print('Test Set: {}, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, val_loss.cpu().numpy()[0], val_error.cpu().numpy()[0]))
        # print('==========================================================================================================')

        now_val_loss = val_loss.item()
        now_val_acc = 1 - val_error.item()

    return model, 0, 0, now_val_loss, now_val_acc


if __name__ == "__main__":

    # Training settings
    now_split = 0
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                        help='weight decay')#10e-5
    parser.add_argument('--target_number', type=int, default=9, metavar='T' ,
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


    validation_loader = data_utils.DataLoader(MILBags(bag_root = '/Data3/GCA_Demo/4864patch_dataset1_bag',
                                                  MIL = True),
                                         batch_size=1,
                                         shuffle=False)

    print('Init Model')
    cluster_num = 1
    model = DeepAttnMIL_Surv(cluster_num=cluster_num)
    MODEL_PATH = glob.glob(os.path.join('results_stage1_clustering_dataset1', '*.pth'))[0]
    save_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    model.load_state_dict(save_dict)
    output_folder = './results_stage1_clustering_dataset1'


    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    print('Start Testing')
    best_model_wts = model.state_dict()
    best_epoch = 0
    best_loss = 10e9
    best_acc = 0.0
    losses = []
    accs = []

    test_losses = []
    test_accs = []
    for epoch in range(0,1):
        model, now_train_loss, now_train_acc, now_test_loss, now_test_acc = test(epoch, model, output_folder)
