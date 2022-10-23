#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/15 21:55
@desc:
"""
import os
import sys
import pandas as pd
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)
import time
import torch
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from lib.BasicTrainer_sw import Trainer
from lib.dataloader import get_dataloader, get_dataloader_stamp
from lib.TrainInits import print_model_parameters
from others.attn_lstm import Attn_LSTM
from others.mGBRT import mGBRT
from others.mLSTM import mLSTM
from others.transformer_bottleneck import DecoderTransformer
from others.informer import Informer
import numpy as np
from lib.dataloader import data_loader
from torch.utils.data import DataLoader


def add_noise2(xx, yy, ratio, flag):
    x_num = xx.shape[0]
    n_num = int(x_num * ratio)
    x_len = xx.shape[1]
    y_len = yy.shape[1]
    noiseData = np.random.normal(0.5, 0.8 ** 2, (n_num, x_len + y_len))
    if flag == 1:
        choice_num = np.random.choice(x_num, n_num)
        noise_stamp_x = xx[choice_num, :, 1:]
        noise_stamp_y = yy[choice_num, :, 1:]
        noise_data_xy = torch.hstack((noise_stamp_x, noise_stamp_y))
        noiseData = noiseData.reshape(n_num, x_len + y_len, 1)
        noiseData = torch.tensor(noiseData)
        noiseData = torch.cat((noiseData, noise_data_xy), 2)

    noiseData = torch.tensor(noiseData).to(args.device)
    noiseData = noiseData.float()
    noiseData_X = noiseData[:, :x_len]
    noiseData_Y = noiseData[:, -y_len:]

    new_X = torch.vstack((xx.to(args.device), noiseData_X))
    new_Y = torch.vstack((yy.to(args.device), noiseData_Y))

    b = torch.randperm(new_X.size(0))
    new_X = new_X[b, :]
    b = torch.randperm(new_Y.size(0))
    new_Y = new_Y[b, :]

    return new_X, new_Y

today = time.time()

for tt in range(3):
    print(f'--------------{tt}--------------------inter----')
    for DATASET in ['powerLoad', 'traffic', 'ETDataset']:  # dataset name  traffic  powerLoad ETDataset

        Mode = 'Train'  # Train or test
        DEBUG = 'True'
        optim = 'sgd'
        DEVICE = 'cuda:0'
        MODEL = 'mLSTM' #Attn_LSTM   gbrt  DecoderTransformer informer  mLSTM
        noise_ratio = 0  # 0 0.3
        ktype = 'normal' #sadc, normal, cagrad, pcgrad
        finish_time = 1632304248.5494814
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
        args.stamp = False
        if MODEL == 'informer':
            args.stamp = True
            args.freq = 'h' #'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]
            args.timeenc = 0
            args.label_len =48
            # args.batch_size = 32

        # load dataset
        if MODEL == 'informer':
            train_loader, val_loader, test_loader, scaler = get_dataloader_stamp(args, normalizer=args.normalizer)
        else:
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

            if args.model == 'Attn_LSTM':
                model = Attn_LSTM(args)
            elif args.model == 'mLSTM':
                model = mLSTM(args)
            elif args.model == 'gbrt':
                model = mGBRT(args)
            elif args.model == 'DecoderTransformer':
                model = DecoderTransformer(args, {'sparse':True, 'attn_pdrop':0.1, 'resid_pdrop':0.1})

            elif args.model == 'informer':
                enc_in = 1
                dec_in = 1
                c_out = 1
                seq_len = args.interval*args.lag
                label_len = args.label_len
                out_len = args.window*args.horizon

                model = Informer(args,
                enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor = 5, d_model = 512, n_heads = 8, e_layers = 3, d_layers = 2, d_ff = 512,
                dropout = 0.0, attn = 'prob', embed = 'fixed', freq = 'h', activation = 'gelu',
                output_attention = False, distil = True, mix = True,
                device = torch.device('cuda:0')
                )

            model = model.to(args.device)
            for p in model.parameters():
                if p.dim() > 1:
                    #nn.init.xavier_uniform_(p)
                    nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(p)

            #print_model_parameters(model, only_num=False)


            #init loss function, optimizer
            if args.loss_func == 'mask_mae':
                loss = masked_mae_loss(scaler, mask_value=0.0)
            elif args.loss_func == 'mae':
                loss = torch.nn.L1Loss().to(args.device)
            elif args.loss_func == 'mse':
                loss = torch.nn.MSELoss().to(args.device)
            else:
                raise ValueError

            ###############add nosie samples##########################

            train_set = train_loader.dataset
            train_set = DataLoader(dataset=train_set, batch_size=1, shuffle=True)
            trainX = []
            trainY = []
            for index, (X, y_true) in enumerate(train_set):
                trainX.append(X.cpu().numpy())
                trainY.append(y_true.cpu().numpy())

            if args.model == 'informer':
                newX, newY = add_noise2(torch.tensor(trainX).squeeze(), torch.tensor(trainY).squeeze(),
                                        args.noise_ratio, 1)
            else:
                newX, newY = add_noise2(torch.tensor(trainX).squeeze(), torch.tensor(trainY).squeeze(),
                                        args.noise_ratio, 0)
            train_dataloader = data_loader(newX, newY, args.batch_size, shuffle=True, drop_last=False)
            ###############add nosie samples##########################

            if args.model not in ['gbrt']:

                optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_init, weight_decay=0)
                #learning rate decay
                lr_scheduler = None
                if args.lr_decay:
                    print('Applying learning rate decay.')
                    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                        milestones=lr_decay_steps,
                                                                        gamma=args.lr_decay_rate)
                #start training
                trainer = Trainer(ktype, model, loss, optimizer, train_dataloader, val_loader, test_loader, scaler,
                                  args, lr_scheduler=lr_scheduler)

                if args.mode == 'Train':
                    trainer.train()

                    if tt == 0:
                        trainer.results.to_csv(
                            f'./results/{args.dataset}_{args.model}_{optim}_{args.early_stop_patience}_ns2_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv',
                            mode='a',
                            header=True
                        )
                    else:
                        trainer.results.to_csv(
                            f'./results/{args.dataset}_{args.model}_{optim}_{args.early_stop_patience}_ns2_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv',
                            mode = 'a',
                            header = False
                        )

                elif args.mode == 'test':
                    model.load_state_dict(torch.load('./experiments/best_model_{}.pth'.format(finish_time)))
                    print("Load saved model")
                    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger, finish_time=finish_time)

            else:
                # start training
                trainer = Trainer(ktype, model, loss, None, train_loader, val_loader, test_loader, scaler,
                                  args, )
                if args.model == 'gbrt':
                    mae, rmse, mape = trainer.test_gbrt(model, args, train_loader, test_loader, scaler, trainer.logger,
                                                        path=None, finish_time=current_time)
                    pf = pd.DataFrame({
                        'dataset': [args.dataset],
                        'MAE': [mae],
                        'RMSE': [rmse],
                        'MAPE': [mape],
                        'steps': [0],
                        'bestModel': [
                            './experiments/best_model_{}_{}_{}_{}.pth'.format(args.model, args.dataset,
                                                                              args.horizon, finish_time)],
                    })

                    if tt == 0:
                        pf.to_csv(
                            f'./results/{args.dataset}_{args.model}_{optim}_{args.early_stop_patience}_ns2_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv',
                            mode='a',
                            header=True
                        )
                    else:
                        pf.to_csv(
                            f'./results/{args.dataset}_{args.model}_{optim}_{args.early_stop_patience}_ns2_{args.noise_ratio}_{args.horizon * args.window}_{today}.csv',
                            mode='a',
                            header=False
                        )


