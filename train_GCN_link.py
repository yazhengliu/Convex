import networkx as nx

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import torch.nn as nn
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
import math
from torch.nn import Parameter
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import Planetoid
import numpy as np
from utils.load_data import SynGraphDataset
import  argparse
import random
import copy
from sklearn.metrics import f1_score
# dataset = Planetoid('data/PlanetoidCora', 'Cora')
# modelname = 'cora'
#
# data = dataset[0]
# print(data)
# data = train_test_split_edges(data,val_ratio=0.2,test_ratio=0.2)
# print(data)
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
    def __init__(self):
        super(Net, self).__init__()
        # self.features= Parameter(torch.Tensor(data.num_nodes,args.features_size),requires_grad=True)

        self.conv1 = GCNConv(data.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.hidden)
        # self.MLP1 = nn.Linear(args.hidden * 2,args.mlp_hidden)
        # self.MLP2 = nn.Linear(args.mlp_hidden, 2)
        self.dropout = args.dropout
        self.linear = nn.Linear(args.hidden * 2, 2,bias=False)




    def encode(self,edgeindex):
        # print(type(data.x))
        #
        # print(data.x.type())


        x = self.conv1(data.x.to(torch.float32), edgeindex)

        x = x.relu()
        # x = F.dropout(x, self.dropout, training=self.training)
        return self.conv2(x, edgeindex)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # print('edge_index',edge_index)
        # print('len',len(pos_edge_index[0])+len(neg_edge_index[1]))
        # print(max(edge_index[0]))
        # print(max(edge_index[1]))
        h = torch.cat([z[edge_index[0]], z[edge_index[1]]],dim=1)
        h = self.linear(h)
        # h=self.MLP1(h)
        # h=h.relu()
        # # h = F.dropout(h, self.dropout, training=self.training)
        # h=self.MLP2(h)

        # h=h.sum(dim=-1)
        # print('h', h.shape)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        # print('logits.shape',logits.shape)
        return h






def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.int, device=device)
    link_labels[:pos_edge_index.size(1)] = 1
    link_labels= link_labels.type(torch.LongTensor)

    return link_labels


def train(edgeindex):
    model.train()
    # neg_edge_index = negative_sampling(
    #     edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes-1,
    #     num_neg_samples=data.train_pos_edge_index.size(1),
    #     force_undirected=True,
    # )
    neg_edge_index = negative_sampling(
        edge_index=edgeindex, num_nodes=data.num_nodes-1,
        num_neg_samples=edgeindex.size(1),
        force_undirected=True,
    )

    # print('neg',neg_edge_index)
    optimizer.zero_grad()

    z = model.encode(edgeindex)
    link_logits = model.decode(z, edgeindex, neg_edge_index) #data.train_pos_edge_index
    link_labels = get_link_labels(edgeindex, neg_edge_index) #data.train_pos_edge_index
    # print(link_logits)
    # print(link_labels)
    loss = F.cross_entropy(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []

    for prefix in ["val", "test"]:
        prob_f1 = []
        prob_auc = []
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode(edgeindex)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        # link_probs = link_logits.sigmoid()
        link_probs=F.softmax(link_logits,dim=1)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        prob_f1.extend(np.argmax( link_probs, axis=1))
        prob_auc.extend(link_probs[:, 1].cpu().numpy())
        # F1=f1_score(link_labels, prob_f1)
        AUC=roc_auc_score(link_labels, prob_auc)

        # print("F1-Score:{:.4f}".format(f1_score(link_labels, prob_f1)))
        # print("AUC:{:.4f}".format(roc_auc_score(link_labels, prob_auc)))
        perfs.append(AUC)
    return perfs


def split_edge(start,end,flag,clear_time):
    edge_index = [[], []]
    if flag == 'year':
        for key, value in clear_time.items():
            if value[0] >= start and value[0] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    if flag == 'month':
        for key, value in clear_time.items():
            if value[1] >= start and value[1] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])

    if flag=='week':
        for key, value in clear_time.items():
            if value[2] >= start and value[2] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    return edge_index
def clear_time(time_dict):
    edge_time = dict()
    for key, value in time_dict.items():
        month = (value.year - 2004) * 12 + value.month
        week = (value.year - 2004) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
if __name__ == "__main__":
    dataset_name: str = 'bitcoinotc'  # bitcoinalpha bitcoinotc
    dataset_dir: str = 'data/'
    dataset = SynGraphDataset(dataset_dir, dataset_name)
    modelname = dataset_name
    data = dataset[0]
    print(data.time_dict)
    print(data.edge_index)
    data = train_test_split_edges(data)
    print(len(data.train_pos_edge_index[0]))
    # print(len(data.train_neg_edge_index[0]))
    print(len(data.val_pos_edge_index[0]))
    print(len(data.val_neg_edge_index[0]))
    print(len(data.test_pos_edge_index[0]))
    print(len(data.test_neg_edge_index[0]))
    # edgeindex_list=copy.deepcopy(data.train_pos_edge_index.tolist())
    # # edgeindex = [[], []]
    # for i in range(data.num_nodes):
    #     edgeindex_list[0].append(i)
    #     edgeindex_list[1].append(i)
    # edgeindex=torch.tensor(edgeindex_list)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=4,
                        help='Number of hidden units.')
    parser.add_argument('--mlp_hidden', type=int, default=4,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--features_size', type=float, default=8,
                        help='Dropout rate (1 - keep probability).')


    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    best_val_perf = test_perf = 0
    for epoch in range(1, args.epochs):
        for month in range(4,11):
            time_dict = data.time_dict
            clear_time_dict = clear_time(time_dict)
            edge_index_old = split_edge(0, month,'month', clear_time_dict)
            for i in range(0, data.num_nodes):
                edge_index_old[0].append(i)
                edge_index_old[1].append(i)
            edgeindex=torch.tensor(edge_index_old)
            train_loss = train(edgeindex)
            val_perf, tmp_test_perf = test()
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_loss, best_val_perf, test_perf))


    torch.save(model.state_dict(), dataset_dir+dataset_name+'/'+dataset_name +'_GCN_model.pth')



