
import numpy as np
import time
import json
import random
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import argparse
import torch.optim as optim
import re
import torch.nn as nn
from nltk.corpus import stopwords
import nltk.stem
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs,string

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: 增加自连接到邻接矩阵
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: 对节点的特征矩阵进行线性变换
        x = x @ self.weight

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        # row, col = edge_index
        # deg = degree(row, size[0], dtype=x_j.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return  x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
class Net_rumor(torch.nn.Module):
    def __init__(self, nhid, nclass,dropout,args):
        super(Net_rumor, self).__init__()
        self.conv1 = GCNConv(nhid*2, nhid)
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)

        self.conv2=GCNConv(nhid, nclass)
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout
        # self.linear = nn.Linear(768, nfeat)
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.glove_embedding, requires_grad=False)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=args.hidden,
                              batch_first=True, num_layers=args.num_layers, bidirectional=True)  # bidirectional=True



    def forward(self, sentence, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        # print(self.conv1(x, edge_index))
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1,edge_weight=edgeweight1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2,edge_weight=edgeweight2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def back(self, x, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        x_0 = self.conv1(x, edge_index_1,edge_weight=edgeweight1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2,edge_weight=edgeweight2)
        return (x_0, x_1)

    def feature(self, sentence):
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return x

    def forward_v2(self, x, edge_index_1, edge_index_2,edgeweight1,edgeweight2):
        # print(self.conv1(x, edge_index))

        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1,edge_weight=edgeweight1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2,edge_weight=edgeweight2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
class Net(torch.nn.Module):
    def __init__(self, nhid, nclass,dropout,args):
        super(Net, self).__init__()
        self.conv1 = GCNConv(nhid*2, nhid)
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)

        self.conv2=GCNConv(nhid, nclass)
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout
        # self.linear = nn.Linear(768, nfeat)
        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.glove_embedding, requires_grad=False)
        self.bilstm = nn.LSTM(input_size=embed_dim, hidden_size=args.hidden,
                              batch_first=True, num_layers=args.num_layers, bidirectional=True)  # bidirectional=True



    def forward(self, sentence, edge_index_1, edge_index_2):
        # print(self.conv1(x, edge_index))
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

    def back(self, x, edge_index_1, edge_index_2):
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1)

    def feature(self, sentence):
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return x

    def forward_v2(self, x, edge_index_1, edge_index_2):
        # print(self.conv1(x, edge_index))

        # x=self.linear(x)
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
# def accuracy(preds, labels):
#     # preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)
def accuracy_list(pred,true):
    correct=0
    for i in range(0,len(pred)):
        if pred[i]==true[i]:
            correct=correct+1
    return correct/len(true)
def train_all(args):
    embeddings_index = {}
    f = codecs.open('data/pheme/glove.840B.300d.txt', encoding='utf-8')
    # nltk.download()
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    jsonPath = f'data/pheme/word_index.json'

    with open(jsonPath, 'r') as f:
        word_index = json.load(f)
    print('word_index success')
    embedding_numpy = np.load("data/pheme/pheme_embedding.npy")
    print('embedding_numpy', embedding_numpy)
    print(' embedding_numpy success')
    embedding_tensor = torch.FloatTensor(embedding_numpy)

    args.glove_embedding=embedding_tensor

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pheme_clean_path = 'data/pheme/pheme_json/'
    files_name = [file.split('.')[0] for file in os.listdir(pheme_clean_path)]
    file_map = dict()
    for i in range(0, len(files_name)):
        file_map[files_name[i]] = i
    # print(file_map)
    idx_list = list(range(max(file_map.values()) + 1))
    file_map_reverse = {value: key for key, value in file_map.items()}
    label_list = []
    for i in range(0, len(idx_list)):
        file_name = file_map_reverse[i]
        jsonPath = f'data/pheme/pheme_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)
        label_list.append(data['label'])
    # print(label_list)
    label_list_tensor = torch.LongTensor(label_list)

    idx_label_0 = []
    idx_label_1 = []
    for i in range(0, len(label_list)):
        if label_list[i] == 0:
            idx_label_0.append(i)
        if label_list[i] == 1:
            idx_label_1.append(i)
    random.shuffle(idx_label_0)
    random.shuffle(idx_label_1)

    # print(idx_list)
    train_ratio = 0.6
    val_ratio = 0.2

    # random.shuffle(idx_list)
    train_list_0 = idx_label_0[0:math.floor(len(idx_label_0) * train_ratio)]

    val_list_0 = idx_label_0[
                 math.floor(len(idx_label_0) * train_ratio):math.floor(len(idx_label_0) * (train_ratio + val_ratio))]
    test_list_0 = idx_label_0[math.floor(len(idx_label_0) * (train_ratio + val_ratio)):]

    train_list_1 = idx_label_1[0:math.floor(len(idx_label_1) * train_ratio)]

    val_list_1 = idx_label_1[
                 math.floor(len(idx_label_1) * train_ratio):math.floor(len(idx_label_1) * (train_ratio + val_ratio))]
    test_list_1 = idx_label_1[math.floor(len(idx_label_1) * (train_ratio + val_ratio)):]
    train_list = train_list_0 + train_list_1
    val_list = val_list_0 + val_list_1
    test_list = test_list_0 + test_list_1
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    print(len(train_list))
    print(len(val_list))
    print(len(test_list))
    #

    model = Net(
        nhid=args.hidden,
        nclass=2,
        dropout=args.dropout, args=args)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0
    early_stop_step = 10
    temp_early_stop_step = 0
    data_prefix = 'pheme'
    label_train = []
    label_train_pred = []

    for train_idx in train_list:
        print('train_index', train_list.index(train_idx))

        for epoch in range(args.epochs):
            output_goal = train(model,train_idx,optimizer,train_list,file_map_reverse,label_list_tensor)
        label_train_pred.append(torch.unsqueeze(output_goal, 0).max(1)[1].item())
        #
        # # print(edges_index)
        label_train.append(label_list_tensor[train_idx])
        # label_train = torch.tensor(label_train)
        # label_train_pred = torch.tensor(label_train_pred)
        # print('label_true',label_train)
        # print('label_pred',label_train_pred)
        acc_train = accuracy_list(label_train_pred, label_train)
        print('acc_train', acc_train)
        if train_list.index(train_idx) % 100 == 0:
            model.eval()
            label_val = []
            label_val_pred = []
            for val_idx in val_list:
                val_index = val_list.index(val_idx)
                if val_index % 500 == 0:
                    print(val_index)
                file_name = file_map_reverse[val_idx]
                jsonPath = f'data/pheme/pheme_json/{file_name}.json'

                with open(jsonPath, 'r') as f:
                    data = json.load(f)
                sentence = np.array(data['intput sentenxe'])
                sentence = torch.tensor(sentence)

                edges_index = data['edges_3']
                # print('edges_index',edges_index)
                edges_index_tensor = torch.tensor(edges_index)
                if len(edges_index_tensor[0]) > 1:
                    output = model(sentence, edges_index_tensor, edges_index_tensor)
                    loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[val_idx].view(-1))
                    # print(loss)
                    # loss_val = loss + loss_val
                    label_val_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())
                    label_val.append(data['label'])

            print('label_val', label_val)
            print('label_val_pred', label_val_pred)

            # label_val = torch.tensor(label_val)
            # label_val_pred = torch.tensor(label_val_pred)
            acc_val = accuracy_list(label_val_pred, label_val)
            print('acc_val', acc_val)

        if train_list.index(train_idx) % 50 == 0:
            temp_acc = test(model,test_list,file_map_reverse,label_list_tensor)
            if temp_acc > best_acc:
                print('save model')
                best_acc = temp_acc
                torch.save(model.state_dict(), f'data/{data_prefix}/' + data_prefix + '_GCN_model.pth')
                temp_early_stop_step = 0
            # else:
            #     temp_early_stop_step += 1
            #     if temp_early_stop_step >= early_stop_step:
            #         print('early stop')
            #         break

    # model.state_dict(), f'data/{data_prefix}/'+data_prefix + '_GCN_model.pth'

    ############test accuracy
    model.load_state_dict(torch.load(f'data/{data_prefix}/'+data_prefix + '_GCN_model.pth'))
    test(model,test_list,file_map_reverse,label_list_tensor)

    label_val_pred = []
    label_val = []

    for val_idx in val_list:
        val_index = val_list.index(val_idx)
        if val_index % 500 == 0:
            print(val_index)
        file_name = file_map_reverse[val_idx]
        jsonPath = f'data/pheme/pheme_json/{file_name}.json'

        with open(jsonPath, 'r') as f:
            data = json.load(f)
        sentence = np.array(data['intput sentenxe'])
        sentence = torch.tensor(sentence)

        edges_index = data['edges_3']
        edges_index_tensor = torch.tensor(edges_index)
        if len(edges_index_tensor[0]) > 1:
            output = model(sentence, edges_index_tensor, edges_index_tensor)
            loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[val_idx].view(-1))
            # print(loss)
            # loss_val = loss + loss_val
            label_val_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())
            label_val.append(data['label'])
    acc_val = accuracy_list(label_val_pred, label_val)
    print('acc_val', acc_val)


def train(model,train_idx,optimizer,train_list,file_map_reverse,label_list_tensor):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    train_index = train_list.index(train_idx)
    # if train_index % 500 == 0:
    #     print(train_index)

    file_name = file_map_reverse[train_idx]
    jsonPath = f'data/pheme/pheme_json/{file_name}.json'

    with open(jsonPath, 'r') as f:
        data = json.load(f)
    sentence = np.array(data['intput sentenxe'])
    sentence = torch.tensor(sentence)

    edges_index = data['edges_3']
    edges_index_tensor = torch.tensor(edges_index)
    # if len(edges_index_tensor[0]) > 1:
        # optimizer.zero_grad()
    output = model(sentence, edges_index_tensor, edges_index_tensor)
    loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[train_idx].view(-1))
    # print(loss)
    loss.backward()

    # loss_train = loss + loss_train
    optimizer.step()
    return output[0]

def test(model,test_list,file_map_reverse,label_list_tensor):
    model.eval()
    loss_test=0
    label_test = []
    label_test_pred = []
    for test_idx in test_list:
        file_name = file_map_reverse[test_idx]
        jsonPath = f'data/pheme/pheme_json/{file_name}.json'

        with open(jsonPath, 'r') as f:
            data = json.load(f)
        sentence = np.array(data['intput sentenxe'])
        sentence = torch.tensor(sentence)

        edges_index = data['edges_3']
        edges_index_tensor = torch.tensor(edges_index)
        if len(edges_index_tensor[0]) > 1:
            output = model(sentence, edges_index_tensor, edges_index_tensor)
            loss = F.cross_entropy(torch.unsqueeze(output[0], 0), label_list_tensor[test_idx].view(-1))
            # print(loss)
            loss_test = loss + loss_test
            # print(torch.unsqueeze(output[0], 0).max(1)[1].item())
            label_test_pred.append(torch.unsqueeze(output[0], 0).max(1)[1].item())

            # print(edges_index)
            label_test.append(data['label'])
    print('label_test', label_test)
    print('label_test_pred',label_test_pred)


    label_test = torch.tensor(label_test)
    label_test_pred = torch.tensor(label_test_pred)
    acc_test=accuracy_list(label_test_pred, label_test)
    print(
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test),
         )
    return acc_test

