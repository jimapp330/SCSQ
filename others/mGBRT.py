#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test model
"""

import torch
from sklearn import ensemble

import warnings
warnings.filterwarnings("ignore")

class mGBRT(torch.nn.Module):
    def __init__(self, args):
        super(mGBRT, self).__init__()
        self.enc_seq_len = args.interval * args.lag
        self.pre_len = args.window * args.horizon

    def train(self, x, y):

        params = {'n_estimators':300, 'max_depth':3, 'min_samples_split':2,
                  'learning_rate':0.01, 'loss':'ls'}

        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(x, y[:,0])

        return clf

    def forward(self, x, clf, pred_one):

        shape = x.shape
        for i in range(self.pre_len):
            pre = clf.predict(x)
            pre = torch.tensor(pre)
            x[:, 0:shape[1]-1] = x[:,1:shape[1]]
            x[:,shape[1]-1] = pre

            pred_one[:,i] = pre

        return pred_one
