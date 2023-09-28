from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from MIL_dataloader_csv import MILBags

import copy
import matplotlib.pyplot as plt

from DeepAttnMISL_model import DeepAttnMIL_Surv

def calculate_loss(Y_prob, Y):
    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

    return neg_log_likelihood

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label, mask) in enumerate(train_loader):
        bag_label = label[0].cuda()
        # # if args.cuda:
        # #     data, bag_label = data.cuda(), bag_label.cuda()
        # data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        graph = [now_data[0].unsqueeze(2).unsqueeze(2) for now_data in data]
        # calculate loss and metrics
        lbl_probs = model(graph, mask.cuda())

        loss = calculate_loss(lbl_probs, bag_label)
        train_loss += loss.data[0]

        lbl_pred = (lbl_probs > 0.5).type(torch.float64)
        error = torch.abs(bag_label - lbl_pred[0])

        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

        #print('Epoch: {}, iteration: {}, loss: {:.4f}, train error: {:.4f}'.format(epoch, batch_idx, loss.item(), error.item()))

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    # print('==========================================================================================================')
    print('\nEpoch_sum: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error.cpu().numpy()[0]))
    # print('==========================================================================================================')

    now_loss = train_loss.item()
    now_acc = 1 - train_error.item()

    return now_loss, now_acc


def test(best_model_wts):
    model.load_state_dict(best_model_wts)

    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label, mask) in enumerate(test_loader):
        bag_label = label[0].cuda()
        graph = [now_data[0].unsqueeze(2).unsqueeze(2) for now_data in data]
        # calculate loss and metrics
        lbl_probs = model(graph, mask.cuda())

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

    now_loss = test_loss.item()
    now_acc = 1 - test_error.item()

    return now_loss, now_acc


if __name__ == "__main__":

    data_split = 10
    final_testing_acc = []
    for ki in range(data_split):
        now_split = ki

        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
        parser.add_argument('--epochs', type=int, default=200, metavar='N',
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

        train_loader = data_utils.DataLoader(MILBags(data_csv_root='/Data2/GCA/AttentionDeepMIL-master/data_split/data_list_%d.csv' % (now_split),
                                                     feature_csv_root='/Data2/GCA/simsiam/feature_cluster_csv',
                                                     cluster_num=8,
                                                     size=(256, 256),
                                                     mean_bag_length=64,
                                                     var_bag_length=4,
                                                     seed=1,
                                                     train=True),
                                             batch_size=1,
                                             shuffle=True)

        test_loader = data_utils.DataLoader(MILBags(data_csv_root='/Data2/GCA/AttentionDeepMIL-master/data_split/data_list_%d.csv' % (now_split),
                                                    feature_csv_root='/Data2/GCA/simsiam/feature_cluster_csv',
                                                    cluster_num=8,
                                                    size=(256, 256),
                                                    mean_bag_length=64,
                                                    var_bag_length=4,
                                                    seed=1,
                                                    train=False),
                                            batch_size=1,
                                            shuffle=False)

        print('Init Model')
        cluster_num = 8
        model = DeepAttnMIL_Surv(cluster_num=cluster_num)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

        print('Start Training')
        best_model_wts = model.state_dict()
        best_acc = 0.0
        losses = []
        accs = []

        test_losses = []
        test_accs = []
        for epoch in range(1, args.epochs + 1):
            now_loss, now_acc = train(epoch)
            losses.append((now_loss))
            accs.append((now_acc))
            now_model_wts = copy.deepcopy(model.state_dict())

            if now_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())

            test_loss, test_acc = test(best_model_wts)
            test_losses.append((test_loss))
            test_accs.append((test_acc))

            plt.plot(losses, label = "Training")
            plt.plot(test_losses, label = "Testing")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss')
            plt.legend()
            plt.savefig('results/%d_TrainingLoss.png' % (now_split))
            plt.clf()

            plt.plot(accs, label = "Training")
            plt.plot(test_accs, label = "Testing")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy')
            plt.legend()
            plt.savefig('results/%d_TrainingAccuracy.png' % (now_split))
            plt.clf()

        print('Final Testing')
        final_test_loss, final_test_acc = test(best_model_wts)
        torch.save(model, 'results/%d_model.pth' % (now_split))

        final_testing_acc.append((final_test_acc))

    print(final_testing_acc)

