#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Federated Learning with Differential Privacy")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="Rounds of training.")
    parser.add_argument('--num_users', type=int, default=100, help="Number of users: K.")
    parser.add_argument('--frac', type=float, default=0.2, help="The fraction of clients: C.")
    parser.add_argument('--bs', type=int, default=1024, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate for training.")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="Learning rate decay each round.")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Model type to be trained (e.g., cnn, mlp).')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'fmnist'],
                        help="Dataset used for training (e.g., mnist, cifar10, fmnist).")
    parser.add_argument('--iid', action='store_true', help='Whether the data distribution among users is i.i.d.')
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes in the dataset.")
    parser.add_argument('--num_channels', type=int, default=1, help="Number of channels in images of the dataset.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for using CPU.")

    # Differential privacy arguments
    parser.add_argument('--dp_mechanism', type=str, default='Gaussian', choices=['no_dp', 'Gaussian', 'Laplace', 'MA'],
                        help='Differential privacy mechanism to be used (e.g., no_dp, Gaussian, Laplace, MA).')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='Differential privacy epsilon, which determines the strength of privacy.')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='Differential privacy delta, which affects the privacy guarantee.')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='Differential privacy clipping parameter, which limits the sensitivity of each user\'s update.')
    parser.add_argument('--dp_sample', type=float, default=1,
                        help='Sampling rate for moment accountant in MA mechanism.')

    parser.add_argument('--serial', action='store_true', help='Enable serial computation to save GPU memory.')
    parser.add_argument('--serial_bs', type=int, default=128, help='Batch size for serial computation.')

    # Experiment group arguments
    parser.add_argument('--group', type=str, default='control', choices=['control', 'fixed_dp', 'dynamic_dp'],
                        help='Experiment group selection: control (no privacy measures), fixed_dp (fixed strength privacy), or dynamic_dp (dynamically adjusted privacy measures).')


    args = parser.parse_args()
    return args
