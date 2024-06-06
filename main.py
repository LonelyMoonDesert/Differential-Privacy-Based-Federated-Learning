#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

from opacus import PrivacyEngine

def setup_privacy_engine(model, args):
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=len(dataset_train),
        alphas=[10, 100],
        noise_multiplier=1.0 if args.group == 'fixed_dp' else 0.5,
        max_grad_norm=1.0,
    )
    return privacy_engine

# # 动态调整差分隐私的参数
# def adjust_privacy_params(privacy_engine, epoch):
#     if epoch < 5:
#         new_noise_multiplier = 0.25
#     elif epoch < 10:
#         new_noise_multiplier = 0.5
#     else:
#         new_noise_multiplier = 1.0
#     privacy_engine.noise_multiplier = new_noise_multiplier
#     privacy_engine.max_grad_norm = max(1.0 - 0.1 * (epoch // 5), 0.5)



if __name__ == '__main__':
    # parse args
    # 设置随机种子，保证实验可重复
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    # 解析输入参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # 根据不同的数据集加载数据并进行用户划分
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    # 对 CIFAR 数据集进行类似处理，并设置不同的变换（增强）策略
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 根据数据集和模型选择，初始化全局模型
    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # 如果使用差分隐私，使用opacus工具包对模型进行封装
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # 复制全局权重
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # 训练准备，选择参与的客户端
    acc_test = []
    if args.serial:
        clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]



    # 定义数据收集变量
    results = {'accuracy': [], 'privacy_loss': []}


    # 进行多轮训练
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]

        max_grad_norm = 1.0  # 默认梯度裁剪强度
        noise_level = 0.1  # 默认噪声水平

        if args.group == 'dynamic_dp':
            # 根据条件动态调整
            if iter < 5:
                max_grad_norm = 0.5
                noise_level = 0.05
            elif iter < 10:
                max_grad_norm = 0.8
                noise_level = 0.08
            else:
                max_grad_norm = 1.0
                noise_level = 0.1


        for idx in idxs_users:
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), args=args, iter=iter)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, acc_t, t_end - t_start))

        results['accuracy'].append(acc_t)

        acc_test.append(acc_t.item())

        # 模型反向推理攻击，量化隐私泄露
        # privacy_loss = model_inversion_attack(net_glob, args)
        # results['privacy_loss'].append(privacy_loss)

    # 训练完成后，保存结果并绘制性能曲线
    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon))



