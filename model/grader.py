#!/usr/bin/env python3
# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import time
import sys
import pickle
from data_splitting import data_split
from syntactic import get_syntactic_features


#idea of building AES
#representation of text, features extracted from the text

class Grader(nn.Module):

    def __init__(self,
                 text2ind, sent2ind, text_embeddings, sent_embeddings,
                 sent, text,
                 sent_num_features, text_num_features, # list of indices to identify the features
                 rnn_layer_size, rnn_layer_number,
                 grade_scale, activation, dropout_rate, bidirectional, cell_type,
                 static=False,
                 **kwargs):

        super(Grader, self).__init__()
        # data
        self.text = text
        self.sent = sent
        self.text2ind = text2ind
        self.sent2ind = sent2ind
        self.text_embeddings = text_embeddings
        self.sent_embeddings = sent_embeddings
        self.sent_num_features = sent_num_features
        self.text_num_features = text_num_features

        # hyperparameters for LSTM models
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_number = rnn_layer_number
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.bidirectioanl = bidirectional
        self.num_directions = 2 if self.bidirectioanl else 1
        self.cell_type = cell_type
        self.grade_scale = grade_scale
        self.levels = 2 if self.sent and self.text else 1
        if bidirectional:
            self.rnn_out_size = 2 * rnn_layer_size
        else:
            self.rnn_out_size = rnn_layer_size
        self.static = static

        self.sent_embed = nn.Embedding.from_pretrained(self.sent_embeddings)
        self.text_embed = nn.Embedding.from_pretrained(self.text_embeddings)

        if self.static:
            self.sent_embed.weight.requires_grad = False
            self.text_embed.weight.requires_grad = False

        if self.cell_type == 'lstm':
            self.sent_rnn = nn.LSTM(
                input_size = self.sent_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first =True,
                bidirectional = self.bidirectioanl
            )
            self.text_rnn = nn.LSTM(
                input_size = self.text_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first = True,
                bidirectional = self.bidirectioanl
            )
        elif self.cell_type == 'gru':
            self.sent_rnn = nn.GRU(
                input_size = self.sent_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first =True,
                bidirectional = self.bidirectioanl
            )
            self.text_rnn = nn.GRU(
                input_size = self.text_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first = True,
                bidirectional = self.bidirectioanl
            )
        elif self.cell_type == 'rnn':
            self.sent_rnn = nn.RNN(
                input_size = self.sent_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first =True,
                bidirectional = self.bidirectioanl
            )
            self.text_rnn = nn.RNN(
                input_size = self.text_embeddings.size(1),
                hidden_size = self.rnn_layer_size,
                num_layers = self.rnn_layer_number,
                batch_first = True,
                bidirectional = self.bidirectioanl
            )

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.rnn_out_size * self.levels, self.grade_scale)



    

        for name, params in self.sent_rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(params)
           

        for name, params in self.text_rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(params)
            if 'bias' in name:
                nn.init.ones_(params)




    def MoT_layer(self, hidden): #hidden: batch, sent, feature
        sent_dim = hidden.size(1)
        out = torch.zeros(hidden.size(0),hidden.size(2))
        for i in range(hidden.size(0)):
            for j in range(hidden.size(2)):
                out[i][j] = hidden[i][:,j].mean()
        return out


    def forward(self, texts): # input shape: (batch_size, text) ([2,3,4]) size(3)

        if self.sent:
            sent_tensor_list = [torch.LongTensor([self.sent2ind[str(s)] for s in t.sentences]) for t in texts]
            sent_tensor = pad_sequence(sent_tensor_list, batch_first=True, padding_value=0)
            # extract sentence features
            sent_features = self.sent_embed(sent_tensor)
            if self.cell_type == 'lstm':
                sent_out, (sent_h, sent_c) = self.sent_rnn(sent_features, None)
            else: # self.cell_type == 'gru':
                sent_out, sent_h = self.sent_rnn(sent_features,None)

            sent_out = self.MoT_layer(sent_out)

        if self.text:
            text_tensor = torch.LongTensor([self.text2ind[t.id] for t in texts])
            text_tensor_list = [torch.LongTensor([i]) for i in text_tensor]
            text_tensor = pad_sequence(text_tensor_list, batch_first=True, padding_value=0)
        # extract text features
            text_features = self.text_embed(text_tensor)
            if self.cell_type == 'lstm':
                text_out, (text_h, text_c) = self.text_rnn(text_features, None)
            elif self.cell_type == 'gru':
                text_out, text_h = self.text_rnn(text_features,None)
            text_out = self.MoT_layer(text_out)

        if self.sent and self.text:
            out = torch.cat((sent_out, text_out),1)
        elif self.sent:
            out = sent_out
        else:
            out = text_out
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc(out)


        return out