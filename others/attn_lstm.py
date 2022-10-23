#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import random
from others.lstm_tf import Encoder
warnings.filterwarnings("ignore")


class Attn_Decoder(nn.Module):
    def __init__(self, out_dim, hid_dim, max_length, n_layers, dropout):
        super(Attn_Decoder, self).__init__()
        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.max_length = max_length

        self.embedding = nn.Linear(self.out_dim, self.hid_dim)
        self.attn = nn.Linear(self.hid_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hid_dim, self.hid_dim, self.n_layers)
        self.out = nn.Linear(self.hid_dim, self.out_dim)
        self.act = nn.ReLU()

    # Encoder-Decoder Attention
    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).unsqueeze(0)  # 3d->2d->fc->3d
        embedded = self.dropout(embedded)  # (1, batch_size, hidden_size)
        attn_weights = F.softmax(self.attn(
            torch.cat((embedded[0], hidden[0]), dim=1)), dim=1)

        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, enc_seq_len)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        # -->(batch_size, 1, hidden_size)
        output = torch.cat((embedded[0], attn_applied[:, 0, :]), 1)
        # -->(batch_size, hidden_size * 2)
        output = self.attn_combine(output).unsqueeze(0)  # -->(1, batch_size, hidden_size)
        output = self.act(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.out(output[0])
        output = self.act(output)

        return output, hidden, cell, attn_weights


class Seq2Seq(torch.nn.Module):

    def __init__(self, args, encoder, attn_decoder):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = attn_decoder
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        self.teacher_forcing = args.teacher_forcing

    def forward(self, x, sampled_target, teacher_forcing_ratio=0, samp_ids=None):


        pre_y_shape = sampled_target.shape

        encoder_inputs = x.unsqueeze(2)  # (batch_size, timestep)
        target_batch = sampled_target.float()
        input_len = encoder_inputs.shape[1]
        batch_size = target_batch.shape[0]
        # print(sampled_target.shape, target_batch.shape)
        target_len = target_batch.shape[1]
        (encoder_hidden, encoder_cell) = self.encoder.init_H_C(batch_size, self.args.device)
        # encoder_outputs = torch.zeros(batch_size, input_len, self.encoder.hid_dim).to(target_batch.device)
        decoder_outputs = torch.zeros(batch_size, target_len, self.decoder.out_dim).to(target_batch.device)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(encoder_inputs, encoder_hidden, encoder_cell)

        decoder_input = encoder_inputs[:, -1, -self.decoder.out_dim:]
        decoder_hidden, decoder_cell = encoder_hidden, encoder_cell

        decoder_attentions = []
        for di in range(0, target_len):
            prediction, decoder_hidden, decoder_cell, attn_weights = self.decoder(decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

            decoder_outputs[:, di, :] = prediction
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target_batch[:, di].unsqueeze(1) if teacher_force else prediction.detach()

            decoder_attentions.append(attn_weights.detach())
            # print(decoder_outputs.shape)
        pre_y = decoder_outputs.reshape(pre_y_shape)
        #return pre_y, attn_weights
        return pre_y, torch.cat(decoder_attentions,dim=1)

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)


class Attn_LSTM(nn.Module):
    def __init__(self, args):
        super(Attn_LSTM, self).__init__()
        self.args = args
        self.enc_seq_len = args.interval * args.lag  # 144
        self.pre_len = args.window * args.horizon  # 18
        self.feature_size = 1
        self.encoder = Encoder(input_dim=self.feature_size,
                               hid_dim=64,
                               n_layers=1,
                               dropout=0.1)
        self.attn_decoder = Attn_Decoder(out_dim=1,
                                         hid_dim=64,
                                         max_length=self.enc_seq_len,
                                         n_layers=1,
                                         dropout=0.1)
        self.seq2seq = Seq2Seq(self.args,
                               encoder=self.encoder,
                               attn_decoder=self.attn_decoder)

    def forward(self, x, sampled_target, teacher_forcing_ratio=0, samp_ids=None):
        pre_y, attn_weights = self.seq2seq(x, sampled_target, teacher_forcing_ratio=0, samp_ids=None)
        return pre_y, attn_weights
