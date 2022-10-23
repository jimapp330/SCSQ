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
import time
import pandas as pd
import copy
from others.attn_lstm import Attn_LSTM
from lib.metrics import All_Metrics
from torch.utils.data import DataLoader
# import keras.backend as K

class CurLearn(object):
    def __init__(self, scaler, bidx, model, args, train_set, test_set, heldout_set, directory=None,
                 model_family='Attn_LSTM', metric='accuracy',
                 **kwargs):
        self.bidx = bidx
        self.scaler = scaler
        self.args = args
        self.heldout_set = heldout_set
        self.train_set = train_set
        self.test_set = test_set
        self.preEpoches = 2
        self.model_family = model_family
        self.metric = metric

        self.directory = directory
        self.batch_size = args.batch_size

        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

        self.model = model.to(args.device)

        self.device = 'cuda'
        self.optimizer = 'sgd'
        if self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_init)

        # init loss function, optimizer
        if args.loss_func == 'mae':
            self.loss_fun = torch.nn.L1Loss().to(self.args.device)
        else:
            self.loss_fun = torch.nn.MSELoss().to(self.args.device)
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


    def currBybootstrapping(self, epoches):
        model_g = Attn_LSTM(self.args).to(self.device)

        optimizer = torch.optim.SGD(model_g.parameters(), lr=self.args.lr_init)

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        for i in range(epoches):
            for index, (X, y_true) in enumerate(self.train_set):

                model_g.train()
                optimizer.zero_grad()

                X = X.to(self.device)
                y_true = y_true.to(self.device)

                y_true = y_true[:, :self.args.window * self.args.horizon]
                label = y_true[:, -self.args.window * self.args.horizon:]

                # Forward pass
                output, _ = model_g(X, y_true, teacher_forcing_ratio=0.)

                loss = loss_fun(output.squeeze().cuda(), label)

                # Backward pass
                loss.backward()
                optimizer.step()

        model_g.eval()
        train_set = self.train_set.dataset
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=1,
                                  shuffle=True)
        self.size_train = self.train_set.batch_size * self.train_set.batch_sampler.__len__()
        scores = []
        reidx = []

        trainX = []
        trainY = []

        for index, (X, y_true) in enumerate(train_loader):
            X = X.to(self.device)
            y_true = y_true.to(self.device)

            y_true = y_true[:, :self.args.window * self.args.horizon]
            label = y_true[:, -self.args.window * self.args.horizon:]

            trainX.append(X.cpu().numpy())
            trainY.append(label.cpu().numpy())

            # Forward pass
            output, _ = model_g(X, y_true, teacher_forcing_ratio=0.)

            loss = loss_fun(output.squeeze().cuda(), label)
            scores.append(loss.detach().cpu().item())
            reidx.append(index)

        # noiseData_Y = torch.tensor(trainY[:30]).squeeze()
        # noiseData_X = torch.tensor(trainX[:30]).squeeze()
        # n_scores = scores
        # n_reidx = reidx
        # n_list = []
        # for ii in range(1,31):
        #     X = noiseData_X[ii-1,:]
        #     b = torch.randperm(X.size(0))
        #     X = X[b]
        #
        #     y_true = noiseData_Y[ii-1,:]
        #     b = torch.randperm(y_true.size(0))
        #     y_true = y_true[b]
        #
        #     X = X.unsqueeze(0).to(self.device)
        #     y_true = y_true.unsqueeze(0).to(self.device)
        #
        #     # Forward pass
        #     output, _ = model_g(X, y_true, teacher_forcing_ratio=0.)
        #
        #     loss = loss_fun(output.squeeze().cuda(), label)
        #     n_scores.append(loss.detach().cpu().item())
        #     n_reidx.append(index+ii)
        #     n_list.append(index+ii)
        #
        newX = torch.tensor(trainX).squeeze()
        newY = torch.tensor(trainY).squeeze()
        self.trainXY = {'x_train':newX,'y_train':newY}

        hardness_score = [reidx, scores]
        hardness_score = np.array(hardness_score).T
        order = hardness_score[hardness_score[:, 1].argsort()]

        # n_hardness_score = [n_reidx, n_scores]
        # n_hardness_score = np.array(n_hardness_score).T
        # n_order = n_hardness_score[n_hardness_score[:, 1].argsort()]

        # Curriculum by Bootstrapping
        return order[:,0].astype(int)

    def order_by_loss(self, dataset, model):
        size_train = len(dataset.y_train)
        scores = model.predict(dataset.x_train)
        hardness_score = scores[list(range(size_train)), dataset.y_train]
        res = np.asarray(sorted(range(len(hardness_score)), key=lambda k: hardness_score[k], reverse=True))
        return res

    # def exponent_decay_lr_generator(self, decay_rate, minimum_lr, batch_to_decay):
    #     cur_lr = None
    #     def exponent_decay_lr(initial_lr, batch):
    #         nonlocal cur_lr
    #         if batch == 0:
    #             cur_lr = initial_lr
    #         if (batch % batch_to_decay) == 0 and batch != 0:
    #             new_lr = cur_lr / decay_rate
    #             cur_lr = max(new_lr, minimum_lr)
    #         return cur_lr
    #     return exponent_decay_lr
    def exponent_decay_lr_generator(self, decay_rate, minimum_lr, batch_to_decay, initial_lr, batch, cur_lr):
        # cur_lr = None
        #def exponent_decay_lr(initial_lr, batch):
        #    nonlocal cur_lr
        if batch == 0:
                cur_lr = initial_lr
        if (batch % batch_to_decay) == 0 and batch != 0:
                new_lr = cur_lr / decay_rate
                cur_lr = max(new_lr, minimum_lr)
        return cur_lr
        #return exponent_decay_lr

    # def data_function_from_input(self, curriculum, batch_size,
    #                              dataset, order, batch_increase,
    #                              increase_amount, starting_percent):
    #     if curriculum == "random":
    #         np.random.shuffle(order)
    #
    #     if curriculum in ["curriculum", "vanilla", "anti", "random"]:
    #         data_function = self.exponent_data_function_generator(dataset, order, batch_increase, increase_amount,
    #                                                          starting_percent, batch_size=batch_size)
    #
    #     else:
    #         print("unsupprted condition (not vanilla/curriculum/random/anti)")
    #         print("got the value:", curriculum)
    #         raise ValueError
    #     return data_function

    # def exponent_data_function_generator(self, dataset, order, batches_to_increase,
    #                                      increase_amount, starting_percent,
    #                                      batch_size=100):
    #
    #     size_data = dataset['x_train'].shape[0]
    #
    #     cur_percent = 1
    #     cur_data_x = dataset['x_train']
    #     cur_data_y = dataset['y_train']
    #
    #     def data_function(x,y,batch):
    #         nonlocal cur_percent, cur_data_x, cur_data_y
    #
    #         if batch % batches_to_increase == 0:
    #             if batch == 0:
    #                 percent = starting_percent
    #             else:
    #                 percent = min(cur_percent * increase_amount, 1)
    #             if percent != cur_percent:
    #                 cur_percent = percent
    #                 data_limit = np.int(np.ceil(size_data * percent))
    #                 new_data = order[:data_limit]
    #                 cur_data_x = dataset['x_train'][new_data]
    #                 cur_data_y = dataset['y_train'][new_data]
    #         return cur_data_x, cur_data_y
    #
    #     return data_function

    def exponent_data_function_generator(self, dataset, order, batches_to_increase,
                                         increase_amount, starting_percent,batch,cur_percent
                                         ):

        size_data = dataset['x_train'].shape[0]

        cur_data_x = dataset['x_train']
        cur_data_y = dataset['y_train']

        #def data_function(x,y,batch):
        #    nonlocal cur_percent, cur_data_x, cur_data_y

        if batch % batches_to_increase == 0:
                if batch == 0:
                    percent = starting_percent
                else:
                    percent = min(cur_percent * increase_amount, 1)
                if percent != cur_percent:
                    cur_percent = percent
                    data_limit = np.int(np.ceil(size_data * percent))
                    new_data = order[:data_limit]
                    cur_data_x = dataset['x_train'][new_data]
                    cur_data_y = dataset['y_train'][new_data]
        return cur_data_x, cur_data_y, cur_percent

        #return data_function


    def run(self):

        '''
        args.lr_decay_rate = 1.5
        args.learning_rate = 0.035
        args.minimal_lr = 1e-4
        args.lr_batch_size = 32
        args.curriculum = 'curriculum'
        args.batch_increase = 32
        args.increase_amount = 1.9
        args.starting_percent = 100/2500
        args.verbose = True
        
        '''
        order = self.currBybootstrapping(self.preEpoches)
        if self.args.curriculum == "anti":
            order = np.flip(order, 0)
        elif self.args.curriculum == "random":
            np.random.shuffle(order)

        ### noise identification
        # newlen = order.shape[0]
        # orig = self.bidx.numpy()
        # neworder = []
        # oridx = []
        # for ii in range(newlen-100, newlen):
        #     oridx.append(orig[order[ii]])
        #     neworder.append(ii)
        # oridx = np.array(oridx)
        # print(sum(oridx > 1632))
        #########################

        ## start expriment
        start_time_all = time.time()
        histories = []

        num_batches = (self.args.epochs * self.size_train) // self.args.batch_size


        # data_function = self.data_function_from_input()

        self.train_model_batches(order,
                                 num_batches,
                                 verbose=self.args.verbose,
                                 batch_size=self.args.batch_size,
                                 initial_lr=self.args.learning_rate)

    def generate_random_batch(self, x, y, batch_size):
        size_data = x.shape[0]
        if size_data < batch_size:
            batch_size = size_data
        cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
        return x[cur_batch_idxs, :], y[cur_batch_idxs, :]

    def basic_lr_scheduler(self,initial_lr, batch):
        return initial_lr

    def basic_data_function(self, x_train, y_train, batch):
        return x_train, y_train

    def train_model_batches(self,order,num_batches, batch_size=100,
                            test_each=50, initial_lr=1e-3,
                            verbose=False):

        start_time = time.time()

        train_epoch_loss = 0
        best_loss = float('inf')
        not_improved_count = 0
        x_train = self.trainXY['x_train']
        y_train = self.trainXY['y_train']
        cur_percent = 1
        cur_lr = self.args.learning_rate
        for batch in range(num_batches):
            if self.args.curriculum in ["curriculum", "vanilla", "anti", "random"]:
                cur_x, cur_y, cur_percent = self.exponent_data_function_generator(self.trainXY, order,
                                                                     self.args.batch_increase,
                                                                     self.args.increase_amount,
                                                                      self.args.starting_percent,
                                                                     batch, cur_percent)

            # cur_lr = lr_scheduler(initial_lr, batch)
            cur_lr = self.exponent_decay_lr_generator(self.args.lr_decay_rate,
                                                      self.args.minimal_lr,
                                                      self.args.lr_batch_size,
                                                      self.args.learning_rate,
                                                      batch, cur_lr)
            # K.set_value(self.model.optimizer.lr, cur_lr)
            self.optimizer.param_groups[0]['lr'] = cur_lr
            data, target = self.generate_random_batch(cur_x, cur_y, self.batch_size)
            target = target[:, :self.args.window * self.args.horizon]
            batch_loss = self.train_on_batch(batch, data, target)

            if self.heldout_set == None:
                val_dataloader = self.test_set
            else:
                val_dataloader = self.heldout_set
            val_epoch_loss = self.val_epoch(batch, val_dataloader)

            if train_epoch_loss > 1e6:
                print('Gradient explosion detected. Ending...')
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    # self.logger.info("Validation performance didn\'t improve for {} epochs. "
                    #                 "Training stops.".format(self.args.early_stop_patience))
                    print("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                '''
                if epoch % self.args.log_step == 0:
                    self.logger.info('*********************************Current best model saved!')
                '''
                best_model = copy.deepcopy(self.model.state_dict())

        finish_time = time.time()
        training_time = finish_time - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        # save the best model to file
        # if not self.args.debug:
        torch.save(best_model, './experiments/best_model_{}_{}_{}_{}.pth'.format(self.args.model, self.args.dataset,
                                                                                 self.args.horizon, finish_time))
        # self.logger.info("Saving current best model to " + self.best_path)

        # test
        self.model.load_state_dict(best_model)
        # self.val_epoch(self.args.epochs, self.test_loader)
        mae, rmse, mape = self.test(self.model, self.args, self.test_set, self.scaler,
                                    finish_time=finish_time)

        pf = pd.DataFrame({
            'dataset': [self.args.dataset],
            'MAE': [mae],
            'RMSE': [rmse],
            'MAPE': [mape],
            'steps': [batch],
            'bestModel': [
                './experiments/best_model_{}_{}_{}_{}_CL.pth'.format(self.args.model, self.args.dataset, self.args.horizon,
                                                                  finish_time)],
        })

        self.results = pf


    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                #data = data[..., :self.args.input_dim]  # data:[B W N 1]  target:[B P N 1]
                #label = target[..., :self.args.output_dim]
                if self.args.stamp:
                    target = target[:, -self.args.window * self.args.horizon:]
                    label = target[:, :, 0]
                else:
                    target = target[:, :self.args.window * self.args.horizon]
                    label = target

                output,_ = self.model(data, target, teacher_forcing_ratio=0.)
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)

                loss = self.loss_fun(output.squeeze().cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        if epoch % self.args.log_step == 0:
            #self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
            print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def train_on_batch(self, batch, data, target):
        self.model.train()
        data = data.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        # data and target shape: B, T, N, F; output shape: B, T, N, F
        output, decoder_attentions = self.model(data, target, teacher_forcing_ratio=0)
        if self.args.real_value:
            label = self.scaler.inverse_transform(target)

        loss = self.loss_fun(output.squeeze().cuda(), target)
        loss.backward()

        # add max grad clipping
        if self.args.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()

            # log information
        if batch % self.args.log_step == 0:
            print('Train batch {}: Loss: {:.6f}'.format(batch, loss.item()))

        return loss.item()

    @staticmethod
    def test(model, args, data_loader, scaler, path=None, finish_time=0):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        total_attentions = []
        total_datas = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # data = data[..., :args.input_dim]
                # label = target[..., :args.output_dim]
                if args.stamp:
                    target = target[:, -args.window * args.horizon:]
                    label = target[:, :, 0]
                else:
                    target = target[:, :args.window * args.horizon]
                    label = target

                output, batch_attention = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
                total_attentions.append(batch_attention)
                total_datas.append(data)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        if args.real_value:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)

        print_time = finish_time
        if not os.path.exists(
                './results/{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.horizon,
                                                     args.window,
                                                     print_time)):
            try:
                os.makedirs(
                    './results/{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.horizon,
                                                         args.window, print_time))
                np.save(
                    './results/{}_{}_{}_{}_{}/{}_true.npy'.format(args.dataset, args.model, args.horizon,
                                                                     args.window, print_time,
                                                                     args.dataset), y_true.cpu().numpy())
                np.save(
                    './results/{}_{}_{}_{}_{}/{}_pred.npy'.format(args.dataset, args.model, args.horizon,
                                                                     args.window, print_time,
                                                                     args.dataset), y_pred.cpu().numpy())
                if args.model in ['shapeSim']:
                    np.save('./results/{}_{}_{}_{}_{}/{}_data.npy'.format(args.dataset, args.model, args.horizon,
                                                                             args.window,
                                                                             print_time, args.dataset),
                            torch.stack(total_datas, dim=0).cpu().numpy())
                    np.save(
                        './results/{}_{}_{}_{}_{}/{}_attention.npy'.format(args.dataset, args.model, args.horizon,
                                                                              args.window,
                                                                              print_time, args.dataset),
                        torch.stack(total_attentions, dim=0).cpu().numpy())
            except:
                print('attention error')
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred[:, t], y_true[:, t],
                                          args.mae_thresh, args.mape_thresh)
            # logger.info("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            #     t + 1, mae, rmse, mape * 100))
            print("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
        mae, rmse, mape = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        # logger.info("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
        #     mae, rmse, mape * 100))
        print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
        if not os.path.exists(
                './results/{}_{}_{}_{}_{}/{}_metrics.npy'.format(args.dataset, args.model, args.horizon,
                                                                    args.window, print_time,
                                                                    args.dataset)):
            np.save('./results/{}_{}_{}_{}_{}/{}_metrics.npy'.format(args.dataset, args.model, args.horizon,
                                                                        args.window, print_time,
                                                                        args.dataset),
                    torch.tensor([mae, rmse, mape]).cpu().numpy())

        return mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()