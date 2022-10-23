#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test models
"""
import torch
import torch.nn as nn
import warnings
import random

warnings.filterwarnings("ignore")

class PCA:
    def __init__(self, in_dim, out_dim=6):
        super(PCA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def low_rank(self, input):
        batch_size = input.shape[0]
        input_len = input.shape[1]
        assert self.in_dim == input.shape[2], "input dim of pca and features must be equal!"
        if self.in_dim > 6:
            pca_input = input[:, :, :self.in_dim - 1]
            u, s, v = torch.pca_lowrank(pca_input.reshape(-1, self.in_dim - 1), q=self.out_dim - 1)

            u = u.reshape(batch_size, input_len, self.out_dim - 1)
            output = torch.cat((u, input[:, :, -1].unsqueeze(-1)), dim=2)
        else:
            output = input
        return output


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, hid_dim)
        self.lstm = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout)
        self.act = nn.ReLU()


    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        embedded = self.act(embedded).transpose(0, 1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output.transpose(0, 1), (hidden, cell)

    def init_H_C(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hid_dim, device=device),
                torch.zeros(1, batch_size, self.hid_dim, device=device))


class Decoder(torch.nn.Module):
    def __init__(self, out_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Linear(out_dim, hid_dim)
        self.lstm = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).unsqueeze(0)
        embedded = self.act(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output[0])

        return prediction, hidden, cell


class Seq2Seq_tf(torch.nn.Module):
    def __init__(self, args):
        super(Seq2Seq_tf, self).__init__()

        self.args = args
        self.feature_size = 12
        self.pca_out_dim = 6
        self.pca = PCA(in_dim=self.feature_size,
                       out_dim=self.pca_out_dim)
        self.encoder = Encoder(input_dim=self.feature_size,
                               hid_dim=64,
                               n_layers=1,
                               dropout=0.1)
        if self.args.use_pca:
            self.encoder = Encoder(input_dim=self.pca_out_dim,
                                   hid_dim=64,
                                   n_layers=1,
                                   dropout=0.1)

        self.decoder = Decoder(out_dim=1,
                               hid_dim=64,
                               n_layers=1,
                               dropout=0.1)

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        self.teacher_forcing = args.teacher_forcing


    def forward(self, x, sampled_target, graph=None, teacher_forcing_ratio=0.5, samp_ids=None):

        pre_y_shape = sampled_target[:, :, :, -1].shape
        if self.args.sample_sensor and samp_ids is not None:
            sampled_x = x[:, samp_ids, :, :]
        else:
            sampled_x = x
        x_batch = sampled_x.reshape(-1, x.shape[-2],
                                    x.shape[-1]).float()  # (batch_size * sensor_num, batch_size, timestep)
        target_batch = sampled_target.reshape(-1, sampled_target.shape[-2], sampled_target.shape[-1]).float()
        input_len = x_batch.shape[1]
        batch_size = target_batch.shape[0]
        # print(sampled_target.shape, target_batch.shape)
        target_len = target_batch.shape[1]
        (encoder_hidden, encoder_cell) = self.encoder.init_H_C(batch_size, self.args.device)
        encoder_outputs = torch.zeros(batch_size, input_len, self.encoder.hid_dim).to(target_batch.device)
        decoder_outputs = torch.zeros(batch_size, target_len, self.decoder.out_dim).to(target_batch.device)
        if self.args.use_pca:
            pca_out = self.pca.low_rank(x_batch)
            output, (encoder_hidden, encoder_cell) = self.encoder(pca_out, encoder_hidden, encoder_cell)
        else:
            output, (encoder_hidden, encoder_cell) = self.encoder(x_batch, encoder_hidden, encoder_cell)

        decoder_input = x_batch[:, -1, -self.decoder.out_dim:]
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell
        for di in range(0, target_len):

            prediction, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            decoder_outputs[:, di, :] = prediction
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_batch[:, di, :] if teacher_force else prediction.detach()
        # print(decoder_outputs.shape)
        pre_y = decoder_outputs.reshape(pre_y_shape)
        return pre_y

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
