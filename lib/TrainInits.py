#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/9/1 22:05
@desc:
"""
import torch
import random
import numpy as np


def init_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt


def init_optim(model, opt):
    '''
    Initialize optimizer
    :param model:
    :param opt:
    :return:
    '''
    return torch.optim.Adam(params=model.parameters(), lr=opt.lr_init)


def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma=opt.lr_scheduler_rate)


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')