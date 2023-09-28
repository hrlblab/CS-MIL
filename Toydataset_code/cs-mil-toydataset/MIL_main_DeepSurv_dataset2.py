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


def train(epoch, model, output_folder, batch_size = 4, mini_batch = 4):

    'do train'

    model.train()
    train_loss = 0.
    train_error = 0.
    cnt = 0
    for batch_idx, (data, label, mask, _) in enumerate(train_loader):

        cnt += 1
        # reset gradients
        if cnt == 1:
            optimizer.zero_grad()

        bag_label = label[0].cuda()
        
        # calculate loss and metrics
        lbl_probs,_,_ = model(data.cuda(), mask.cuda())

        if cnt == 1:
            loss = calculate_loss(lbl_probs, bag_label)
        else:
            loss += calculate_loss(lbl_probs, bag_label)

        lbl_pred = (lbl_probs > 0.5).type(torch.float64)
        error = torch.abs(bag_label - lbl_pred[0])

        train_error += error

        if cnt == mini_batch or batch_idx == len(train_loader) - 1:
            train_loss += loss.data[0]

            # backward pass
            loss.backward()
            # step
            optimizer.step()
            cnt = 0

        #print('Epoch: {}, iteration: {}, loss: {:.4f}, train error: {:.4f}'.format(epoch, batch_idx, loss.item(), error.item()))

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    # print('==========================================================================================================')
    print('\nEpoch_sum: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error.cpu().numpy()[0]))
    # print('==========================================================================================================')

    now_train_loss = train_loss.item()
    now_train_acc = 1 - train_error.item()


    'do validation'
    model.train()
    val_loss = 0.
    val_error = 0.
    
    df = pd.DataFrame(columns=['bag_num','root','instance_score','20X','10X','5X','class','score'])
    
    with torch.no_grad():
        for batch_idx, (data, label, mask, root) in enumerate(validation_loader):
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
                #if scale_attention[ki].max() < 0:
                   #scale_attention[ki] = torch.abs(scale_attention[ki])

                #cross_scores = softmax(scale_attention[ki])
                cross_scores = scale_attention[ki]
                now_20X = cross_scores[0, 0].item()
                now_10X = cross_scores[0, 1].item()
                now_5X = cross_scores[0, 2].item()
                
                row = len(df)
                df.loc[row] = [now_bag_num, now_root, now_instance_score, now_20X, now_10X, now_5X, bag_label.item(), lbl_probs[0].item()]

        df.to_csv(os.path.join(output_folder,'attention_score_%d.csv' % (int(epoch))), index = False)
        val_error /= len(validation_loader)
        val_loss /= len(validation_loader)

        # print('==========================================================================================================')
        print('Test Set: {}, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, val_loss.cpu().numpy()[0], val_error.cpu().numpy()[0]))
        # print('==========================================================================================================')

        now_val_loss = val_loss.item()
        now_val_acc = 1 - val_error.item()

    return model, now_train_loss, now_train_acc, now_val_loss, now_val_acc


def test(model, best_model_wts, batch_size = 4):
    model.load_state_dict(best_model_wts)

    model.eval()
    test_loss = 0.
    test_error = 0.
    with torch.no_grad():
        for batch_idx, (data, label, mask) in enumerate(test_loader):

            # calculate loss and metrics
            lbl_probs = model(data.cuda(), mask.cuda())

            loss = calculate_loss(lbl_probs, bag_label)
            test_loss += loss.data[0]

            lbl_pred = (lbl_probs > 0.5).type(torch.float64)
            error = torch.abs(bag_label - lbl_pred[0])

            test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    # print('==========================================================================================================')
    print('Test Set: {}, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss.cpu().numpy()[0], test_error.cpu().numpy()[0]))
    # print('==========================================================================================================')

    df = pd.DataFrame(columns=[''])

    now_loss = test_loss.item()
    now_acc = 1 - test_error.item()

    return model, now_loss, now_acc

if __name__ == "__main__":

    # Training settings
    now_split = 0
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
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

    train_loader = data_utils.DataLoader(MILBags(bag_root = '/Data2/GCA_Demo/Dataset2_Bag_1000_8_1/Train',
                                                  MIL = True),
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=8)


    validation_loader = data_utils.DataLoader(MILBags(bag_root = '/Data2/GCA_Demo/Dataset2_Bag_1000_8_1/Val',
                                                  MIL = True),
                                         batch_size=1,
                                         shuffle=False,
                                        num_workers=8)

    print('Init Model')
    cluster_num = 1
    model = DeepAttnMIL_Surv(cluster_num=cluster_num)
    output_folder = 'results_stage1_clustering_dataset2_0.00005'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    #
    MODEL_PATH = glob.glob(os.path.join('results_stage1_clustering_dataset2', '*.pth'))[0]
    save_dict = torch.load(MODEL_PATH, map_location='cpu')['state_dict']
    model.load_state_dict(save_dict)

    print('Start Training')
    best_model_wts = model.state_dict()
    best_epoch = 0
    best_loss = 10e9
    best_acc = 0.0
    losses = []
    accs = []

    test_losses = []
    test_accs = []
    for epoch in range(1, args.epochs + 1):
        model, now_train_loss, now_train_acc, now_test_loss, now_test_acc = train(epoch, model, output_folder)
        losses.append((now_train_loss))
        accs.append((now_train_acc))
        test_losses.append((now_test_loss))
        test_accs.append((now_test_acc))
        now_model_wts = copy.deepcopy(model.state_dict())

        # if (now_test_acc >= best_acc) and (now_test_loss <= best_loss):
        if now_test_loss <= best_loss:
            best_acc = now_test_acc
            best_loss = now_test_loss
            print('yes best acc')
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        plt.plot(losses, label = "Training")
        plt.plot(test_losses, label = "Testing")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_folder,'%d_TrainingLoss.png' % (now_split)))
        plt.clf()

        plt.plot(accs, label = "Training")
        plt.plot(test_accs, label = "Testing")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_folder,'%d_TrainingAccuracy.png' % (now_split)))
        plt.clf()


    # print('Final Testing')
    # best_model, final_test_loss, final_test_acc = test(model, best_model_wts)
    #
    # df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
    # df.loc[0] = [best_epoch, final_test_loss, final_test_acc]
    # df.to_csv(os.path.join(output_folder,'%d_FinalResults.csv' % (now_split)))
    #
    best_model = model
    best_model.load_state_dict(best_model_wts)

    torch.save({
        'epoch': args.epochs + 1,
        'state_dict': best_model.state_dict()
    }, os.path.join(output_folder,'%d_model_%d.pth' % (now_split, best_epoch)))
    # # torch.save(model, 'results/%d_model.pth' % (now_split))
    #
        #final_testing_acc.append((final_test_acc))

#print(test_accs)

