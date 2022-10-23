#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
from others.attn_lstm import Attn_LSTM
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import MNIST
import torch.nn as nn
import  numpy as np
from torchvision.transforms import transforms
from torchvision import  datasets

print('dd')

def return_model(mode, args,**kwargs):
    if mode=='Attn_LSTM':
        model = Attn_LSTM(args)
        return model


class mnist_modi(Dataset):
    def __init__(self,root,download,train,transform):
        self.mnist = datasets.MNIST(root=root,
                                        download=download,
                                        train=train,
                                        transform=transform)
    def __getitem__(self, index):
        data, target = self.mnist[index]
        return data, target, index
    def __len__(self):
        return len(self.mnist)

def crete_data(name,train_size,num_test,batch_size):
    if name=="mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset = mnist_modi(root='./data', train=True, transform=transform, download=True)
        val_set = mnist_modi(root='./data', train=False, transform=transform, download=True)
        indices = np.random.randint(0,59999,(1,train_size))
        total_range= [i for i in range(60000)]
        train_index = [i for i in indices[0,:]]
        heldout_index = list(set(total_range).difference(set(train_index)))

        train_set = torch.utils.data.Subset(dataset, train_index)
        heldout_set = torch.utils.data.Subset(dataset, heldout_index)
        return  train_set,val_set,heldout_set,dataset


def error(mem):
    # if len(mem) < 4:
    #     return 1.0
    # all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-4:]
    # errors = np.mean(np.abs(all_vals[-4:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
    #return np.max(errors)
    return len(mem)

def training(args, model, train_loader, device,learning_rate):
    '''
    Function for the training step of the training loop
    '''

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    model.train()
    running_loss = 0

    if args.loss_func == 'mae':
        loss_fun = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_fun = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    for index, (X, y_true) in enumerate(train_loader):
        optimizer.zero_grad()

        X = X.to(device)
        model = model.to(device)
        y_true = y_true.to(device)

        # Forward pass
        if args.model == 'Autoformer':
            y_true = y_true[:, :args.label_len + args.window * args.horizon]
        else:
            y_true = y_true[:, :args.window * args.horizon]

        if args.model in ['informer', 'Autoformer']:
            label = y_true[:, -args.window * args.horizon:, 0]
        else:
            label = y_true[:, -args.window * args.horizon:]

        output, _ = model(X, y_true, teacher_forcing_ratio=0.)

        loss = loss_fun(output.squeeze().cuda(), label)

        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def prediction_cost():
    pass
def score(args, model,test_set,device):
    # test_loader = DataLoader(dataset=test_set,
    #                               batch_size=32,
    #                               shuffle=False)
    total_loss = 0
    n = 0

    # init loss function, optimizer
    if args.loss_func == 'mae':
        loss_fun = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_fun = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    with torch.no_grad():
        model.eval()
        for index, (X, y_true) in enumerate(test_set):
            X = X.to(device)
            model = model.to(device)
            y_true = y_true.to(device)

            if args.model == 'Autoformer':
                y_true = y_true[:, :args.label_len + args.window * args.horizon]
            else:
                y_true = y_true[:, :args.window * args.horizon]

            if args.model in ['informer', 'Autoformer']:
                label = y_true[:, -args.window * args.horizon:, 0]
            else:
                label = y_true[:, -args.window * args.horizon:]

            output, _ = model(X, y_true, teacher_forcing_ratio=0.)

            loss = loss_fun(output.squeeze().cuda(), label)

            # a whole batch of Metr_LA is filtered
            if not torch.isnan(loss):
                total_loss += loss.item()
            else:
                print('nan')

    val_loss = total_loss / len(test_set)

    return val_loss

def predict_proba():
        pass
def train_gshap(args, model,train_loader,test_set,criterion,optimizer,learning_rate,device):
    vals=[]
    indexes=[]
    if optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    running_loss = 0

    # init loss function, optimizer
    if args.loss_func == 'mae':
        loss_fun = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_fun = torch.nn.MSELoss().to(args.device)
    else:
        raise ValueError

    for index, (X, y_true) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        X = X.to(device)
        model = model.to(device)
        y_true = y_true.to(device)

        if args.model == 'Autoformer':
            y_true = y_true[:, :args.label_len + args.window * args.horizon]
        else:
            y_true = y_true[:, :args.window * args.horizon]

        if args.model in ['informer', 'Autoformer']:
            label = y_true[:, -args.window * args.horizon:, 0]
        else:
            label = y_true[:, -args.window * args.horizon:]

        # Forward pass
        output, _ = model(X, y_true, teacher_forcing_ratio=0.)

        loss = loss_fun(output.squeeze().cuda(), label)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        vals.append(score(args,model,test_set,device))
        indexes.append(index)
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, np.array(vals),np.array(indexes)

def fit_gshape(args, model,train_set,test_set,max_epochs,criterion,optimizer,learning_rate,device):
    # train_loader = DataLoader(dataset=train_set,
    #                               batch_size=1,
    #                               shuffle=True)

    history = {'metrics': [], 'idxs': []}
    for epoch in range(0, max_epochs):
        model, vals_metrics, idxs = train_gshap(args, model, train_set ,test_set, criterion, optimizer,learning_rate, device)
        history['idxs'].append(idxs)
        history['metrics'].append(vals_metrics)
    return history

def fit(args, train_loader,model,max_epochs,device,learning_rate):
    # train_loader = DataLoader(dataset=train_set,
    #                           batch_size=32,
    #                           shuffle=True)
    train_losses = []
    # Train model
    for epoch in range(0, max_epochs):
            # training
        model, optimizer1, train_loss = training(args, model,train_loader , device,learning_rate)
        train_losses.append(train_loss)
    return model, optimizer1, train_losses

