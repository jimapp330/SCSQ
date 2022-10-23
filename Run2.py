#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/15 21:55
@desc:
"""
import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import argparse
import configparser
from datetime import datetime
import torch.nn as nn
from others.attn_lstm import Attn_LSTM
from lib.dataloader import get_dataloader
from lib.Power_CL import CurLearn
from lib.DShap_p import DShap
from lib.sgd_influence import SGDInfluence
import time

today = time.time()
from lib.dataloader import data_loader
from torch.utils.data import DataLoader

def add_noise2(xx, yy, ratio):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    x_len = xx.shape[1]
    y_len = yy.shape[1]
    noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
    noiseData = torch.tensor(noiseData).to(args.device)
    noiseData = noiseData.float()
    noiseData_X = noiseData[:, :x_len]
    noiseData_Y = noiseData[:, -y_len:]

    new_X = torch.vstack((xx.to(args.device), noiseData_X))
    new_Y = torch.vstack((yy.to(args.device), noiseData_Y))

    b = torch.randperm(new_X.size(0))
    new_X = new_X[b, :]
    # bb = torch.randperm(new_Y.size(0))
    new_Y = new_Y[b, :]

    return new_X, new_Y, b

for tt in range(3):

    print(f'--------------{tt}--------------------inter----')
    for DATASET in ['ETDataset','traffic', 'powerLoad', 'ETDataset']:  # dataset name  traffic  powerLoad ETDataset

        Mode = 'Train'  # Train or test
        DEBUG = 'True'

        DEVICE = 'cuda:0'
        MODEL = 'Attn_LSTM'
        ktype = 'sgd-influence'  # sgd-influence, d-shapely, cl
        noise_ratio = 0  # 0 0.3


        finish_time = 1662287958.9541638
        # 获取配置文件
        config_file = 'configs/{}.conf'.format(DATASET)
        config = configparser.ConfigParser()
        config.read(config_file)

        from lib.metrics import MAE_torch
        def masked_mae_loss(scaler, mask_value):
            def loss(preds, labels):
                if scaler:
                    preds = scaler.inverse_transform(preds)
                    labels = scaler.inverse_transform(labels)
                mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
                return mae
            return loss

        #parser
        args = argparse.ArgumentParser(description='arguments')
        args.add_argument('--dataset', default=DATASET, type=str)
        args.add_argument('--mode', default=Mode, type=str)
        args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
        args.add_argument('--debug', default=DEBUG, type=eval)
        args.add_argument('--model', default=MODEL, type=str)
        args.add_argument('--cuda', default=True, type=bool)
        #data
        args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
        args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
        args.add_argument('--lag', default=config['data']['lag'], type=int)
        args.add_argument('--step', default=config['data']['step'], type=int)
        args.add_argument('--window', default=config['data']['window'], type=int)
        args.add_argument('--interval', default=config['data']['interval'], type=int)
        args.add_argument('--horizon', default=config['data']['horizon'], type=int)
        args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
        args.add_argument('--stamp', default=config['data']['stamp'], type=bool)
        #model
        args.add_argument('--en_input_dim', default=config['model']['en_input_dim'], type=int)
        args.add_argument('--de_input_dim', default=config['model']['de_input_dim'], type=int)
        args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
        args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
        args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
        args.add_argument('--layer_size', default=config['model']['layer_size'], type=int)
        args.add_argument('--res_channels', default=config['model']['res_channels'], type=int)
        args.add_argument('--skip_channels', default=config['model']['skip_channels'], type=int)
        args.add_argument('--column_wise', default=config['model']['column_wise'], type=bool)

        #train
        args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
        args.add_argument('--seed', default=config['train']['seed'], type=int)
        args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
        args.add_argument('--epochs', default=config['train']['epochs'], type=int)
        args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
        args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
        args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
        args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
        args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
        args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
        args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
        args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
        args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=eval)
        args.add_argument('--tf_decay_steps', default=config['train']['tf_decay_steps'], type=int, help='teacher forcing decay steps')
        args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
        #test
        args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
        args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
        #log
        args.add_argument('--log_dir', default='./', type=str)
        args.add_argument('--log_step', default=config['log']['log_step'], type=int)
        args.add_argument('--plot', default=config['log']['plot'], type=eval)

        args = args.parse_args()

        args.noise_ratio = noise_ratio
        #for cl
        if ktype == 'cl':
            args.batch_size = 32
        #for dshapely
        if ktype == 'd-shapely':
            args.batch_size = 1
        if ktype == 'sgd-influence':
            args.batch_size = 32

        train_loader, val_loader, test_loader, scaler = get_dataloader(args, normalizer=args.normalizer)

        for hh in [1,2,4]:
            args.horizon = hh
            args.window = 24
            #init_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.set_device(int(args.device[5]))
            else:
                args.device = 'cpu'

            #config log path
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            current_dir = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
            args.log_dir = log_dir

            #init model
            args.stamp = False

            model = Attn_LSTM(args)
            for p in model.parameters():
                if p.dim() > 1:
                    #nn.init.xavier_uniform_(p)
                    nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(p)

            directory = './temp'


            if ktype == 'cl':
                args.lr_decay_rate = 1.5
                args.learning_rate = 0.03
                args.minimal_lr = 1e-4
                args.lr_batch_size = 40
                args.curriculum = 'curriculum'
                args.batch_increase = 100
                args.increase_amount = 1.5
                args.starting_percent = 100 / 2500
                args.verbose = True
                args.early_stop_patience = 10


            ###############add nosie samples##########################
            # args.noise_ratio = 0.3
            train_set = train_loader.dataset
            train_set = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
            trainX = []
            trainY = []
            for index, (X, y_true) in enumerate(train_set):
                trainX.append(X.cpu().numpy())
                trainY.append(y_true.cpu().numpy())

            newX, newY, bidx = add_noise2(torch.tensor(trainX).squeeze(), torch.tensor(trainY).squeeze(),
                                         args.noise_ratio)

            train_dataloader = data_loader(newX, newY, args.batch_size, shuffle=False, drop_last=False)
            ###############add nosie samples##########################

            if ktype == 'cl':
                quality = CurLearn(scaler, bidx,
                                 model, args, train_dataloader, test_loader, val_loader, directory=directory,
                                 model_family=args.model, metric='accuracy')

                quality.run()
                train_res = quality.results

            elif ktype == 'sgd-influence':
                quality = SGDInfluence(
                    model, args, train_dataloader, test_loader, val_loader, directory=directory,
                    model_family=args.model, metric='accuracy')

                quality.run(1, sdg_run=True)
                # Delete the ones with large values
                sorted_vals = np.argsort(quality.vals_g)[::-1]

                ### noise identification
                # newlen = sorted_vals.shape[0]
                # orig = bidx.numpy()
                # neworder = []
                # oridx = []
                # for ii in range(100):
                #     oridx.append(orig[sorted_vals[ii]])
                #     neworder.append(ii)
                # oridx = np.array(oridx)
                # print(sum(oridx > 1632))
                ###########################

                train_res = quality._portion_performance_series(sorted_vals, scaler)


            elif ktype == 'd-shapely':
                directory = './temp'
                quality = DShap(model, args, train_dataloader, test_loader, val_loader, 1000,
                                sources=None,
                                sample_weight=None,
                                model_family=args.model,
                                metric='accuracy',
                                overwrite=True,
                                directory=directory, seed=0)

                quality.run(1, 1, g_run=True)
                # Delete the ones with small values
                sorted_vals = np.argsort(quality.vals_g)

                ### noise identification
                # newlen = sorted_vals.shape[0]
                # orig = bidx.numpy()
                # neworder = []
                # oridx = []
                # for ii in range(100):
                #     oridx.append(orig[sorted_vals[ii]])
                #     neworder.append(ii)
                # oridx = np.array(oridx)
                # print(sum(oridx > 1632))
                #########################

                train_res = quality._portion_performance_series(sorted_vals, scaler)


            if args.noise_ratio == 0:
                file_path = f'./results/{args.dataset}_{args.model}_{ktype}_{args.horizon * args.window}_{today}.csv'
            else:
                file_path = f'./results/{args.dataset}_{args.model}_{ktype}_ns2_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv'

            if tt == 0:
                train_res.to_csv(
                    file_path,
                    mode='a',
                    header=True
                )
            else:
                train_res.to_csv(
                    file_path,
                    mode='a',
                    header=False
                )


