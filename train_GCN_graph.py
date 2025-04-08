import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
import math
import numpy as np
import torch.nn as nn

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

class GCN(torch.nn.Module):
    def __init__(self, nfeat,hidden_channels,nclass):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        # self.weight=Parameter(torch.Tensor(nfeat, hidden_channels),requires_grad=True)
        self.conv1 = GCNConv(nfeat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, nclass)
        # self.embedding=nn.Embedding(40, hidden_channels)
        # self.lin = Linear(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index, batch):
        # 1. 获得节点嵌入
        # x=[i for i in range(x.shape[0])]
        #
        # x=self.embedding(torch.tensor(x))


        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        # print(batch)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print('x',x.shape)

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin(x)
        return x

    def back(self, x, edge_index_1, edge_index_2):

        x_0 = self.conv1(x, edge_index_1)
        x_1 = F.relu(x_0)
        x = self.conv2(x_1, edge_index_2)
        return (x_0, x_1, x)

    def forward_pre(self, x, edge_index_1, edge_index_2):
        # print('shuru',x)
        # x = torch.topk(x, 1)[1].squeeze(1)
        # print('x', x)
        # x = self.embedding(x)
        # print('x', x)
        x = F.relu(self.conv1(x, edge_index_1))
        # print('x[goal]',x[goal])

        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index_2)

        # x = global_mean_pool(x,batch)
        # print(x)

        # x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x=self.conv3(x,edge_index)
        return x
    def get_feature(self,x):
        # x = [i for i in range(x.shape[0])]
        # x = self.embedding(torch.tensor(x))
        return x



def train():
    model.train()
    optimizer.zero_grad()
    loss=0
    for i in range(150):
        data=dataset[i]
        batch=torch.tensor([0]*(data.x.shape[0]))
        out = model(data.x, data.edge_index, batch)
        loss += criterion(out, data.y)
    loss.backward()
    optimizer.step()



    # for data in train_loader:
    #     optimizer.zero_grad()
    #
    #     out = model(data.x, data.edge_index, data.batch)
    #     loss = criterion(out, data.y)
    #
    #     loss.backward()
    #     optimizer.step()



def acc_val():
    model.eval()

    correct = 0

    for i in range(150,180):
        data=dataset[i]
        batch = torch.tensor([0] * (data.x.shape[0]))
        out = model(data.x, data.edge_index, batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct/30

def acc_train():
    model.eval()

    correct = 0

    for i in range(150):
        data=dataset[i]
        batch = torch.tensor([0] * (data.x.shape[0]))
        out = model(data.x, data.edge_index, batch)  # 一次前向传播
        pred = out.argmax(dim=1)  # 使用概率最高的类别
        correct += int((pred == data.y).sum())  # 检查真实标签
    return correct/150


    # for data in loader:  # 批遍历测试集数据集。
    #     out = model(data.x, data.edge_index, data.batch)  # 一次前向传播
    #     pred = out.argmax(dim=1)  # 使用概率最高的类别
    #     correct += int((pred == data.y).sum())  # 检查真实标签
    # return correct / len(loader.dataset)


if __name__=='__main__':
    dataset = TUDataset('data/TUDataset', name='MUTAG',use_node_attr='True')

    # print(data.x)
    # print(data.edge_index)
    # print(data.edge_attr)

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(dataset.num_node_labels)
    print(dataset.num_node_features)
    print(dataset[0])


    dataset = dataset.shuffle()


    train_dataset = dataset[:100]
    test_dataset = dataset[100:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print(data.batch)
        print(data.x)
    print(dataset.num_node_features)
    model = GCN(nfeat=7,hidden_channels=16,nclass=2)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, 200):
        train()
        train_acc = acc_train()
        test_acc = acc_val()
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    torch.save(model.state_dict(), 'data/TUDataset/GCN_model.pth')

    model.eval()
    model.load_state_dict(torch.load('data/' + 'TUdataset/' + 'GCN_model' + '.pth'))

    # print(dataset[0].x)
    # print(torch.nonzero(dataset[0].x).squeeze())
    #
    # print()
    # y= torch.topk(dataset[0].x, 1)[1].squeeze(1)
    # num_embeddings=100
    # embed_dim=16
    # z=nn.Embedding(num_embeddings, embed_dim,)
    # output = z(y)
    # print(output)





