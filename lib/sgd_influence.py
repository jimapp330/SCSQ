#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import os
import torch
import numpy as np
import _pickle as pkl
import copy
import joblib
from lib.BasicTrainer_sw import Trainer
from others.attn_lstm import Attn_LSTM
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class SGDInfluence(object):
    def __init__(self, model, args, train_set, test_set, heldout_set, directory=None,
                 model_family='Attn_LSTM', metric='accuracy',
                 **kwargs):

        self.args = args
        self.heldout_set = heldout_set
        self.train_set = train_set
        self.learning_rate = 0.001
        self.test_set = test_set

        self.max_epochs = 1
        self.model_family = model_family
        self.metric = metric

        self.directory = directory
        self.batch_size = args.batch_size
        self.args.topk = 100
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

            self._initialize_instance(train_set)

        self.model = model.to(args.device)

        self.device = 'cuda'
        self.optimizer = 'sgd'



    def _initialize_instance(self, train_set):

        n_sources = len(train_set) * self.batch_size
        n_points = len(train_set) * self.batch_size
        self.sgd_nmbrs = self._which_parallel(self.directory)
        self._create_results_placeholder(self.directory, self.sgd_nmbrs, n_points, n_sources)

    def _create_results_placeholder(self, directory, sgd_number, n_points, n_sources):
        # sgd_dir = os.path.join(
        #     directory,
        #     'sgd_{}.pkl'.format(sgd_number.zfill(4))
        # )

        self.mem_sgd = np.zeros((0, n_points))
        self.idxs_sgd = np.zeros((0, n_sources), int)

        # pkl.dump({'mem_sgd': self.mem_sgd}, open(sgd_dir, 'wb'))

    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        sgd_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'sgd' in name]

        sgd_nmbrs = str(np.max(sgd_nmbrs) + 1) if len(sgd_nmbrs) else '0'

        return sgd_nmbrs


    def add_noise2(self, xx, yy, ratio):
        x_num = xx.shape[0]
        n_num = int(x_num * ratio)
        x_len = xx.shape[1]
        y_len = yy.shape[1]
        noiseData = np.random.normal(0.5, 0.8**2, (n_num, x_len+y_len))
        noiseData = torch.tensor(noiseData).to(self.device)
        noiseData = noiseData.float()
        noiseData_X = noiseData[:,:x_len]
        noiseData_Y = noiseData[:,-y_len:]

        new_X = torch.vstack((xx.to(self.device), noiseData_X))
        new_Y = torch.vstack((yy.to(self.device), noiseData_Y))

        b = torch.randperm(new_X.size(0))
        new_X = new_X[b,:]
        b = torch.randperm(new_Y.size(0))
        new_Y = new_Y[b,:]

        return new_X, new_Y
    def add_noise(self, xx, yy, ratio):
        x_num = xx.shape[0]
        n_num = int(x_num * ratio)
        x_len = xx.shape[1]
        y_len = yy.shape[1]
        noiseData = np.random.random((n_num, x_len+y_len))
        noiseData = torch.tensor(noiseData).to(self.device)
        noiseData = noiseData.float()
        noiseData_X = noiseData[:,:x_len]
        noiseData_Y = noiseData[:,-y_len:]

        new_X = torch.vstack((xx.to(self.device), noiseData_X))
        new_Y = torch.vstack((yy.to(self.device), noiseData_Y))

        b = torch.randperm(new_X.size(0))
        new_X = new_X[b,:]
        b = torch.randperm(new_Y.size(0))
        new_Y = new_Y[b,:]

        return new_X, new_Y


    def run(self, counts, sdg_run=False):

        while sdg_run:

            if len(self.mem_sgd) >= counts:
                sdg_run = False
            else:
                self._sgd_inf()
                self.vals_g = np.mean(self.mem_sgd, 0)

        # if self.directory is not None:
        #     self.save_results()

    def _sgd_inf(self):
        iterations = self.max_epochs
        for iteration in range(iterations):

            train_data, list_of_models = self.pre_fit()

            model_g = Attn_LSTM(self.args).to(self.device)
            model_g.load_state_dict(copy.deepcopy(self.model.state_dict()))

            print('{} out of {} G-Shapley iterations'.format(
                iteration + 1, iterations))
            marginal_contribs = self.fit_sgd(iteration, model_g, train_data, list_of_models)

            self.mem_sgd = np.concatenate(
                [self.mem_sgd, np.reshape(marginal_contribs.detach().cpu().numpy(), (1, -1))])

    def pre_fit(self):

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        train_data = {}

        # model list
        bundle_size = len(self.train_set)
        list_of_models = [Attn_LSTM(self.args).to(self.device) for _ in range(bundle_size)]

        for index, (X, y_true) in enumerate(self.train_set):
            # save
            list_of_models[index].load_state_dict(copy.deepcopy(self.model.state_dict()))

            train_data[index] = (X, y_true)

            self.model.train()
            optimizer.zero_grad()

            X = X.to(self.device)
            self.model = self.model.to(self.device)
            y_true = y_true.to(self.device)

            y_true = y_true[:, :self.args.window * self.args.horizon]
            label = y_true[:, -self.args.window * self.args.horizon:]

            # Forward pass
            output, _ = self.model(X, y_true, teacher_forcing_ratio=0.)

            loss = loss_fun(output.squeeze().cuda(), label)

            # Backward pass
            loss.backward()
            optimizer.step()

        return train_data, list_of_models

    def fit_sgd(self, epoch, model_g, train_data, list_of_models):

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        lr = self.learning_rate
        device = self.device
        model_g.eval()
        # gradient
        u = self.compute_gradient(model_g)

        ntr = len(self.train_set.dataset)
        keys = train_data.keys()

        # influence
        infl = torch.zeros(ntr, 1, requires_grad=False).to(device)

        for k in range(len(keys)):
            m = list_of_models[k].to(device)

            # influence
            idx_train = train_data[k]
            x_train = idx_train[0]
            y_true = idx_train[1]

            y_true = y_true[:, :self.args.window * self.args.horizon]
            label = y_true[:, -self.args.window * self.args.horizon:]

            x_size = x_train.shape[0]
            for i in range(x_size):
                s_id = k * self.batch_size + i
                # Forward pass
                output, _ = m(x_train[i].unsqueeze(0), y_true[i].unsqueeze(0), teacher_forcing_ratio=0.)

                loss = loss_fun(output.squeeze().cuda(), label[i])

                m.zero_grad()
                loss.backward()
                for j, param in enumerate(m.parameters()):
                    infl[s_id, 0] += lr * (u[j].data * param.grad.data).sum() / x_size

            # update u
            with torch.backends.cudnn.flags(enabled=False):
                output, _ = m(x_train, y_true, teacher_forcing_ratio=0.)
            loss = loss_fun(output.squeeze().cuda(), label)

            grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
            ug = 0
            for uu, g in zip(u, grad_params):
                ug += (uu * g).sum()
            m.zero_grad()
            ug.backward()
            for j, param in enumerate(m.parameters()):
                u[j] -= lr * param.grad.data / x_size

        # save
        fn = '%s/infl_sgd_at_epoch_%02d.dat' % (self.directory, epoch)
        joblib.dump(infl.cpu().numpy(), fn, compress=9)

        # if epoch > 0:
        #     infl[:, epoch - 1] = infl[:, epoch].clone()

        return infl

    def compute_gradient(self, model):
        # 在val_dataset
        device = self.device
        n = len(self.heldout_set) * self.batch_size

        u = [torch.zeros(*param.shape, requires_grad=False).to(device) for param in model.parameters()]
        model.train()

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for index, (X, y_true) in enumerate(self.heldout_set):
            y_true = y_true[:, :self.args.window * self.args.horizon]
            label = y_true[:, -self.args.window * self.args.horizon:]
            # Forward pass
            output, _ = model(X, y_true, teacher_forcing_ratio=0.)

            loss = loss_fun(output.squeeze().cuda(), label)
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            for j, param in enumerate(model.parameters()):
                try:
                    u[j] += param.grad.data / n
                except (Exception):
                    u[j] += param.data / n
        return u

    def _portion_performance_series(self, idxs, scaler):
        """Given a set of indexes, starts removing points from
        the first elemnt and evaluates the new model after
        removing each point."""

        top_k = idxs[:self.args.topk]
        heldout_index = list(set(idxs).difference(set(top_k)))
        #heldout_index = np.array(top_k)
        train_set = self.train_set.dataset
        data = torch.utils.data.Subset(train_set, heldout_index)
        train_loader = DataLoader(dataset=data,
                                  batch_size=32,
                                  shuffle=True)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.args.lr_init, weight_decay=0)
        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        # learning rate decay
        lr_scheduler = None
        if self.args.lr_decay:
            print('Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(self.args.lr_decay_step.split(','))]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                                milestones=lr_decay_steps,
                                                                gamma=self.args.lr_decay_rate)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)

        # start training
        val_loader = DataLoader(dataset=self.heldout_set.dataset,
                                batch_size=32,
                                shuffle=True)

        trainer = Trainer('normal',self.model, loss, optimizer, train_loader, val_loader, self.test_set, scaler,
                          self.args, lr_scheduler=lr_scheduler)

        if self.args.mode == 'Train':
            trainer.train()
            return trainer.results
