import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
import math
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        
        self.nfeat = nfeat
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear_r = nn.Linear(5, nfeat)
        self.linear_p = nn.Linear(6, nfeat)
        self.linear_u = nn.Linear(7, nfeat)

    def forward(self, inputx, adj, nums):
        x = torch.zeros(len(inputx),self.nfeat)
#        for it, k in enumerate(inputx):
#            if len(k) == 5:
#                x[it] = self.linear_r(torch.FloatTensor(k))
#            elif len(k) == 6:
#                x[it] = self.linear_p(torch.FloatTensor(k))
#            else:
#                x[it] = self.linear_u(torch.FloatTensor(k))
        x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        if nums[1][0] != nums[1][1]:
            x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
            x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        if nums[2][0] != nums[2][1]:
            x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
            x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

    def output_layer(self, inputx, adj,is_cuda):
        x = torch.zeros(len(inputx), self.nfeat)
        for it, k in enumerate(inputx):
            k = k.type(torch.FloatTensor)
            if is_cuda:
                k = k.cuda()
            if len(k) == 5:
                x[it] = self.linear_r(k)
            elif len(k) == 6:
                x[it] = self.linear_p(k)
            else:
                x[it] = self.linear_u(k)
        #x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        #x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        #if nums[1][0] != nums[1][1]:
        #    x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
        #    x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        #if nums[2][0] != nums[2][1]:
        #    x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
        #    x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        #x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))
        if is_cuda:
            x = x.cuda()
        return F.relu(self.gc1(x, adj))
    def feature(self,inputx,nums):
        x = torch.zeros(len(inputx), self.nfeat)
        #        for it, k in enumerate(inputx):
        #            if len(k) == 5:
        #                x[it] = self.linear_r(torch.FloatTensor(k))
        #            elif len(k) == 6:
        #                x[it] = self.linear_p(torch.FloatTensor(k))
        #            else:
        #                x[it] = self.linear_u(torch.FloatTensor(k))
        x[:nums[0][0]] = self.linear_r(torch.FloatTensor(inputx[:nums[0][0]]))
        x[nums[0][0]:nums[0][1]] = self.linear_u(torch.FloatTensor(inputx[nums[0][0]:nums[0][1]]))
        if nums[1][0] != nums[1][1]:
            x[nums[0][1]:nums[1][0]] = self.linear_r(torch.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
            x[nums[1][0]:nums[1][1]] = self.linear_u(torch.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        if nums[2][0] != nums[2][1]:
            x[nums[1][1]:nums[2][0]] = self.linear_r(torch.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
            x[nums[2][0]:nums[2][1]] = self.linear_u(torch.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        x[nums[2][1]:] = self.linear_p(torch.FloatTensor(inputx[nums[2][1]:]))
        return x
    def back(self,features, adj):
        x_0=self.gc1(features, adj)
        x_1 = F.relu(x_0)
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        x = self.gc2(x_1, adj)
        return (x_0,x_1,x)
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
class Net(torch.nn.Module):
    def __init__(self, nfeat,nhid, nclass,dropout):
        super(Net, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid )
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)

        self.conv2=GCNConv(nhid, nclass )
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout




    def forward(self, x, edge_index_1,edge_index_2):
        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)
    def back(self,x, edge_index_1,edge_index_2):
        x_0=self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0,x_1,x)
class Net2(torch.nn.Module):
    def __init__(self, nfeat,nhid, nclass,dropout):
        super(Net2, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid )
        self.conv2 = GCNConv(nhid, nclass )

        # self.conv2=GCNConv(nhid1, nhid2 )
        # self.conv3 = GCNConv(nhid2, dataset.num_classes)
        # self.conv4 = GCNConv(nhid3, dataset.num_classes)
        self.dropout = dropout




    def forward(self, x, edge_index):
        # print(self.conv1(x, edge_index))
        x = F.relu(self.conv1(x, edge_index))

        # x = F.dropout(x, self.dropout,training=self.training)
        x = self.conv2(x, edge_index)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
        # return F.log_softmax(x, dim=1)

class Net_rumor(torch.nn.Module):
    def __init__(self, nhid, nclass,dropout,args):
        super(Net_rumor, self).__init__()
        self.conv1 = GCNConv(nhid*2, nhid )
        # self.conv2 = GCNConv(nhid1, dataset.num_classes)

        self.conv2=GCNConv(nhid, nclass )
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
    def feature(self,sentence):
        x = self.embed(sentence)
        # print(x)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        _, (hidden, cell) = self.bilstm(x)
        # x=x[-1, :, :]
        x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return x
    def back_gcn(self,x, edge_index_1, edge_index_2):
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return x

    def back(self, x, edge_index_1, edge_index_2):
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1, x)
class Net_link(torch.nn.Module):
    def __init__(self,nfeat,nhid):
        super(Net_link, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.linear = nn.Linear(nhid * 2, 2,bias=False)
        # self.MLP1 = nn.Linear(args.hidden * 2, args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)

    def encode(self, x,edge_index):
        # print(type(data.x))
        #
        # print(data.x.type())

        x = self.conv1(x.to(torch.float32), edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h=self.linear(h)
        # h = self.MLP1(h)
        # h = h.relu()
        # h = self.MLP2(h)

        # h=h.sum(dim=-1)
        # print('h', h.shape)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # print('logits.shape',logits.shape)
        return h

    def back_MLP(self, z, pos_edge_index):
        edge_index = pos_edge_index
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        h = self.linear(h)
        return h
        # h_0 = self.MLP1(h)
        # h_1 = h_0.relu()
        # h_2 = self.MLP2(h_1)
        # return h_0, h_1, h_2

    def back_GCN(self, x, edge_index_1, edge_index_2):
        x.to(torch.float32)
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1, x)

    def back_1(self, x, edge_index_1, edge_index_2):
        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        # h = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        # h = self.MLP1(h)
        # h = h.relu()
        # h = self.MLP2(h)
        return x

    def back_2(self, h):
        h = self.linear(h)
        return h
        # h_0 = self.MLP1(h)
        # h_1 = h_0.relu()
        # h_2 = self.MLP2(h_1)
        # return h_2
