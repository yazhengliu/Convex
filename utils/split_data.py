import pickle
from .load_data import *
from .utils_deeplift import *
import argparse
from  .GCN_models import GCN,Net,Net2,Net_rumor,Net_link
import torch.optim as optim
import os
import json
from .load_data import SynGraphDataset
import random
class gen_Yelp_data():
    def __init__(self,dataset,start1,end1,start2,end2,flag):
        self.dataset=dataset
        self.start1=start1
        self.end1=end1
        self.start2=start2
        self.end2=end2
        self.flag=flag
    def gen_adj(self,):
        target_domain = self.dataset  # test
        source_domain = self.dataset
        data_prefix = 'data/'
        with open(data_prefix + target_domain + '_features.pickle', 'rb') as f:
            raw_features = pickle.load(f)

        with open(data_prefix + 'ground_truth_' + target_domain, 'rb') as f:
            review_ground_truth = pickle.load(f)

        with open(data_prefix + 'messages_' + target_domain, 'rb') as f:
            messages = pickle.load(f)

        with open(data_prefix + f'{self.dataset}_split_data.pickle', 'rb') as f:
            rev_time = pickle.load(f)
        train_ratio=0.5
        val_ratio=0.2
        train_rev = read_data('train', 'review', target_domain, train_ratio, val_ratio)
        val_rev = read_data('val', 'review', target_domain, train_ratio, val_ratio)
        test_rev = read_data('test', 'review', target_domain, train_ratio, val_ratio)
        # print('train_rev',train_rev)
        train_user, train_prod = read_user_prod(train_rev)
        val_user, val_prod = read_user_prod(val_rev)
        test_user, test_prod = read_user_prod(test_rev)
        # print('train_user',train_user)

        portion_train = train_rev + train_user
        portion_val = val_rev + val_user
        portion_test = test_rev + test_user
        # print('portion_train',portion_train)
        list_idx, features, nums = feature_matrix(raw_features, portion_train, portion_val, portion_test)

        labels, user_ground_truth = onehot_label(review_ground_truth, list_idx)
        print(labels)
        idx_map = {j: i for i, j in enumerate(list_idx)}
        user_list = []
        prod_list = []
        rev_list = []
        for key, value in idx_map.items():
            if isinstance(key, tuple):
                rev_list.append(value)
            if key[0] == 'u':
                user_list.append(value)
            if key[0] == 'p':
                prod_list.append(value)
        adj_old = construct_adj_matrix(review_ground_truth, idx_map, labels, rev_time, self.start1, self.end1, self.flag)
        adj_new = construct_adj_matrix(review_ground_truth, idx_map, labels, rev_time, self.start2, self.end2, self.flag)
        edges_old = construct_edge(review_ground_truth, idx_map, labels, rev_time, self.start1, self.end1, self.flag)
        edges_new = construct_edge(review_ground_truth, idx_map, labels, rev_time, self.start2, self.end2, self.flag)
        adj_old = adj_old + sp.eye(adj_old.shape[0])
        adj_new = adj_new + sp.eye(adj_new.shape[0])
        adj_old_nonzero = adj_old.nonzero()
        adj_new_nonzero = adj_new.nonzero()
        addedgelist, removeedgelist = difference(edges_new, edges_old)
        addedgelist = clear(addedgelist)
        # print(addedgelist)
        removeedgelist = clear(removeedgelist)



        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)
        return adj_old,adj_new,edges_old,edges_new,graph_old,graph_new,addedgelist,removeedgelist,features,nums
    def gen_parameters(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--dataset', type=str, default='Chi')
        parser.add_argument('--train_ratio', type=float, default=0.5)
        parser.add_argument('--val_ratio', type=float, default=0.2)

        args = parser.parse_args()

        model = GCN(nfeat=32,
                    nhid=args.hidden,
                    nclass=2,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        model.eval()

        model.load_state_dict(torch.load('data/'+ '70_dynamic_GCN_model_in_' + self.dataset + '.pth'))
        adj_old,adj_new,_,_,_,_,_,_,features,nums=self.gen_adj()
        features_clear = model.feature(features, nums)
        model.eval()
        adj_old = sparse_mx_to_torch_sparse_tensor(adj_old)
        output_old = model(features, adj_old, nums)
        layernumbers = 2

        Hold = dict()
        Hold[0] = features_clear.detach().numpy()
        Hold[1], Hold[2], Hold[3] = model.back(features_clear, adj_old)
        Hold[1] = Hold[1].detach().numpy()
        Hold[2] = Hold[2].detach().numpy()
        Hold[3] = Hold[3].detach().numpy()

        adj_new = sparse_mx_to_torch_sparse_tensor(adj_new)
        model.eval()
        output_new = model(features, adj_new, nums)
        Hnew = dict()
        Hnew[0] = features_clear.detach().numpy()
        Hnew[1], Hnew[2], Hnew[3] = model.back(features_clear, adj_new)
        Hnew[1] = Hnew[1].detach().numpy()
        Hnew[2] = Hnew[2].detach().numpy()
        Hnew[3] = Hnew[3].detach().numpy()

        W = dict()
        model.eval()
        W1 = model.state_dict()['gc1.weight']
        W2 = model.state_dict()['gc2.weight']

        W[0] = features_clear.detach().numpy()
        W[1] = W1.detach().numpy()
        W[2] = W2.detach().numpy()

        model_mask = Net(nfeat=32,
                         nhid=args.hidden,
                         nclass=2,
                         dropout=args.dropout)
        model_mask.eval()

        model_dict = model_mask.state_dict()

        model_dict['conv1.weight'] = model.state_dict()['gc1.weight']
        model_dict['conv2.weight'] = model.state_dict()['gc2.weight']
        # model_dict['conv3.weight'] = pretrained_dict['gc3.weight']
        model_mask.load_state_dict(model_dict)

        model_gnn = Net2(nfeat=32,
                         nhid=args.hidden,
                         nclass=2,
                         dropout=args.dropout)
        model_gnn.eval()


        model_gnn.load_state_dict(model_dict)
        return Hold,Hnew,W,features_clear,model_mask,model_gnn
class gen_rumor_data():
    def __init__(self, dataset,data_path,embedding_path,model_path,start,end):
        self.dataset=dataset
        self.data_path=data_path
        self.embedding_path=embedding_path
        self.model_path=model_path
        self.start=start
        self.end=end
    def gen_idxlist(self):
        files_name = [file.split('.')[0] for file in os.listdir(self.data_path)]
        file_map = dict()
        for i in range(0, len(files_name)):
            file_map[files_name[i]] = i
        # print(file_map)
        idx_list = list(range(max(file_map.values()) + 1))
        file_map_reverse = {value: key for key, value in file_map.items()}
        return idx_list,file_map,file_map_reverse
    def gen_adj(self,file_index):
        _,file_map,file_map_reverse=self.gen_idxlist()
        file_name = file_map_reverse[file_index]
        jsonPath = f'data/{self.dataset}/{self.dataset}_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)


        # x = np.array(data['x'])
        edges_old = data[self.start]  # 0,2/3
        edges_new = data[self.end]  # 1/3,1
        adj_old = rumor_construct_adj_matrix(edges_old, len(data['node_map']))
        adj_new = rumor_construct_adj_matrix(edges_new, len(data['node_map']))
        adj_old_nonzero = adj_old.nonzero()
        adj_new_nonzero = adj_new.nonzero()

        addedgelist, removeedgelist = difference(edges_new, edges_old)
        # print(addedgelist)
        addedgelist = clear(addedgelist)
        # print(addedgelist)
        removeedgelist = clear(removeedgelist)
        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)
        return adj_old,adj_new,edges_old,edges_new,graph_old,graph_new,addedgelist,removeedgelist





    def gen_parameters(self,file_index,model,edges_new,edges_old,file_map_reverse):
        # _, file_map, file_map_reverse = self.gen_idxlist()
        file_name = file_map_reverse[file_index]
        jsonPath = f'data/{self.dataset}/{self.dataset}_json/{file_name}.json'
        with open(jsonPath, 'r') as f:
            data = json.load(f)

        edges_new_tensor = torch.tensor(edges_new)
        edges_old_tensor = torch.tensor(edges_old)


        sentence = np.array(data['intput sentenxe'])
        sentence = torch.LongTensor(sentence)
        # print('sentence',sentence)
        model.eval()
        x_tensor = model.feature(sentence)
        x = x_tensor.detach().numpy()
        Hold = dict()
        Hold[0] = x

        Hold[1], Hold[2], Hold[3] = model.back(x_tensor, edges_old_tensor, edges_old_tensor)
        Hold[1] = Hold[1].detach().numpy()
        Hold[2] = Hold[2].detach().numpy()
        Hold[3] = Hold[3].detach().numpy()

        Hnew = dict()
        Hnew[0] = x
        Hnew[1], Hnew[2], Hnew[3] = model.back(x_tensor, edges_new_tensor, edges_new_tensor)
        Hnew[1] = Hnew[1].detach().numpy()
        Hnew[2] = Hnew[2].detach().numpy()
        Hnew[3] = Hnew[3].detach().numpy()

        W = dict()
        model.eval()
        W1 = model.state_dict()['conv1.weight']
        W2 = model.state_dict()['conv2.weight']

        W[0] = x
        W[1] = W1.detach().numpy()
        W[2] = W2.detach().numpy()


        return Hold, Hnew, W, x
    def gen_model(self):
        embedding_numpy = np.load(self.embedding_path, allow_pickle=True)
        print('embedding_numpy success')
        embedding_tensor = torch.FloatTensor(embedding_numpy)

        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--epochs', type=int, default=200,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--glove_embedding', type=float, default=embedding_tensor,
                            )
        parser.add_argument('--num_layers', type=int, default=2,
                            )

        args = parser.parse_args()

        model = Net_rumor(
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout, args=args)
        model.eval()
        model.load_state_dict(torch.load(self.model_path))

        model_gnn = Net2(nfeat=args.hidden * 2,
                         nhid=args.hidden,
                         nclass=2,
                         dropout=args.dropout)
        model_gnn.eval()

        model_dict = model_gnn.state_dict()

        model_dict['conv1.weight'] = model.state_dict()['conv1.weight']
        model_dict['conv2.weight'] = model.state_dict()['conv2.weight']
        model_gnn.load_state_dict(model_dict)
        return model,model_gnn
class gen_link_data():
    def __init__(self, dataset,data_path,start1,end1,start2,end2,flag):
        self.dataset=dataset
        self.data_path=data_path
        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.flag = flag

    def load_data(self):
        dataset = SynGraphDataset(self.data_path,self.dataset)
        modelname = self.dataset
        data = dataset[0]
        time_dict = data.time_dict
        if self.dataset=='UCI':
            clear_time_dict = clear_time_UCI(time_dict)
        else:
            clear_time_dict = clear_time(time_dict)
        edge_index_old = split_edge(self.start1, self.end1, self.flag, clear_time_dict)
        edge_index_new = split_edge(self.start2, self.end2, self.flag, clear_time_dict)
        for i in range(0, data.num_nodes):
            edge_index_old[0].append(i)
            edge_index_old[1].append(i)
            edge_index_new[0].append(i)
            edge_index_new[1].append(i)
        adj_old = rumor_construct_adj_matrix(edge_index_old, data.num_nodes)
        adj_new = rumor_construct_adj_matrix(edge_index_new, data.num_nodes)
        addedgelist, removeedgelist = difference(edge_index_new, edge_index_old)
        # print(addedgelist)
        addedgelist = clear(addedgelist)
        # print(addedgelist)
        removeedgelist = clear(removeedgelist)
        adj_old_nonzero = adj_old.nonzero()
        adj_new_nonzero = adj_new.nonzero()
        graph_old = matrixtodict(adj_old_nonzero)
        graph_new = matrixtodict(adj_new_nonzero)
        edge_index_all = copy.deepcopy(edge_index_new)
        for remove_edge in removeedgelist:
            edge_index_all[0].append(remove_edge[0])
            edge_index_all[1].append(remove_edge[1])
            edge_index_all[1].append(remove_edge[0])
            edge_index_all[0].append(remove_edge[1])
        return dataset,adj_old, adj_new, edge_index_old, edge_index_new, edge_index_all,graph_old, graph_new, addedgelist, removeedgelist
    def gen_model(self,data):
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
        parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--data.x_size', type=float, default=8,
                            help='Dropout rate (1 - keep probability).')

        args = parser.parse_args()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model= Net_link(data.num_features,args.hidden).to(device)
        model.eval()
        data_prefix = self.data_path+'/'+self.dataset+'/'+self.dataset
        model.load_state_dict(torch.load(data_prefix+ '_GCN_model.pth'))

        return model

    def gen_parameters(self,model, edge_index_all):
        model.eval()
        dataset,adj_old, adj_new, edge_index_old, edge_index_new, \
        edge_index_all, graph_old, graph_new, addedgelist, removeedgelist=self.load_data()
        Hold_mlp = dict()

        edge_index_old_tensor=torch.tensor(edge_index_old)
        encode_logits_old = model.encode(dataset[0].x,edge_index_old_tensor)
        decode_logits_old = model.decode(encode_logits_old, edge_index_all)
        Hold_mlp[1] = model.back_MLP(encode_logits_old, edge_index_all)
        Hold_mlp[1] = Hold_mlp[1].detach().numpy()

        model.eval()
        Hnew_mlp = dict()
        edge_index_new_tensor = torch.tensor(edge_index_new)
        encode_logits_new = model.encode(dataset[0].x,edge_index_new_tensor)
        decode_logits_new = model.decode(encode_logits_new, edge_index_all)
        decode_label_new = decode_logits_new.argmax(dim=1)

        Hnew_mlp[1] = model.back_MLP(encode_logits_new, edge_index_all)
        Hnew_mlp[1] = Hnew_mlp[1].detach().numpy()

        W_mlp = dict()
        model.eval()
        W1 = model.state_dict()['linear.weight']
        # W2 = model.state_dict()['linear.bias']

        W_mlp[1] = W1.detach().numpy().T
        # W_mlp[2] = W2.detach().numpy()

        Hold = dict()
        data=dataset[0]
        Hold[0] = data.x.numpy()
        # Hold[1] = activation['gc1']
        # Hold[2] = F.relu(Hold[1])
        # Hold[3] = activation['gc2']
        Hold[1], Hold[2], Hold[3] = model.back_GCN(data.x.to(torch.float32), edge_index_old_tensor,
                                                   edge_index_old_tensor)
        Hold[1] = Hold[1].detach().numpy()
        Hold[2] = Hold[2].detach().numpy()
        Hold[3] = Hold[3].detach().numpy()

        model.eval()
        # output_new = model(x_tensor, edges_new_tensor, edges_new_tensor)
        Hnew = dict()
        Hnew[0] = data.x.numpy()
        Hnew[1], Hnew[2], Hnew[3] = model.back_GCN(data.x.to(torch.float32), edge_index_new_tensor,
                                                   edge_index_new_tensor)
        Hnew[1] = Hnew[1].detach().numpy()
        Hnew[2] = Hnew[2].detach().numpy()
        Hnew[3] = Hnew[3].detach().numpy()

        W = dict()
        model.eval()
        W1 = model.state_dict()['conv1.weight']
        W2 = model.state_dict()['conv2.weight']

        W[0] = data.x
        W[1] = W1.detach().numpy()
        W[2] = W2.detach().numpy()

        return Hold,Hnew,W,W_mlp,Hnew_mlp,Hold_mlp,decode_logits_new,decode_logits_old,decode_label_new
class gen_graph_data():
    def __init__(self,dataset,index,addedgenum,removeedgenum):
        self.dataset=dataset
        self.index=index
        self.addedgenum=addedgenum
        self.removeedgenum=removeedgenum
    def gen_original_edge(self):
        data = self.dataset[self.index]
        # print(data)
        #
        # print(len(data.edge_index[0]))

        x, edge_index = data.x, data.edge_index

        edge_index_list = data.edge_index.numpy().tolist()
        for i in range(0, x.shape[0]):
            edge_index_list[0].append(i)
            edge_index_list[1].append(i)
        edge_index = torch.tensor(edge_index_list)
        adj_old = rumor_construct_adj_matrix(edge_index_list, x.shape[0])
        # print(adj_old)
        adj_old_nonzero = adj_old.nonzero()
        graph_old = matrixtodict(adj_old_nonzero)
        edges_dict_old = dict()
        for i, node in enumerate(edge_index_list[0]):
            edges_dict_old[(node, edge_index_list[1][i])] = i

        return x,edge_index,graph_old,edges_dict_old,adj_old
    def random_edges(self,edge_index,graph_old,edges_dict_old):
        edge_index_list = edge_index.numpy().tolist()
        edge_index_list_new = copy.deepcopy(edge_index_list)
        random.seed()
        removeedgelist = []
        removeedgeindex = []
        removeedgesave = []
        addedgelist = []

        addedgesave = []
        data = self.dataset[self.index]
        x=data.x
        # addedgelist = [[13,6]]
        # removeedgelist=[[12,13]]
        # for path in removeedgelist:
        #     removeedgeindex.append(edges_dict_old[(path[0], path[1])])
        #     removeedgeindex.append(edges_dict_old[(path[1], path[0])])


        for i in range(0, self.removeedgenum):
            a = random.choice(list(range(x.shape[0])))
            # print('a',a)
            # print(graph_old[a])
            list1 = copy.deepcopy(graph_old[a])
            list1.remove(a)

            if list1 != []:
                b = random.choice(list1)
                # print(retD)
                if [a,b] not in removeedgelist and [b,a] not in removeedgelist:
                    removeedgelist.append([a, b])
                    # removeedgesave.append((str(a), str(b)))
                    removeedgeindex.append(edges_dict_old[(a, b)])
                    removeedgeindex.append(edges_dict_old[(b, a)])


        for i in range(0,self.addedgenum):

            c = random.choice(list(range(x.shape[0])))
            retD = list(set(range(0, x.shape[0])).difference(set(graph_old[c])))
            # print(retD)
            if retD != []:
                d = random.choice(retD)
                if [c,d] not in addedgelist and [d,c] not in addedgelist:
                    addedgelist.append([c, d])
                    addedgesave.append((str(c), str(d)))



        # print(len(removeedgeindex))
        # print('removeedgelist', removeedgelist)
        removeedgeindex = list(set(removeedgeindex))
        removeedgeindex = sorted(removeedgeindex)


        for j in reversed(removeedgeindex):
            # if [edge_index_list_new[0][j], edge_index_list_new[1][j]] in removeedgelist or [edge_index_list_new[1][j], edge_index_list_new[0][j]] in removeedgelist:
            #     print('yes')
            # else:
            #     print('false')
            # print((edge_index_list_new[0][j], edge_index_list_new[1][j]))
            del edge_index_list_new[0][j]
            del edge_index_list_new[1][j]
        # print('len edge',len(edge_index_list_new[0]))
        for addpath in addedgelist:
            edge_index_list_new[0].append(addpath[0])
            edge_index_list_new[1].append(addpath[1])
            edge_index_list_new[1].append(addpath[0])
            edge_index_list_new[0].append(addpath[1])
        adj_new = rumor_construct_adj_matrix(edge_index_list_new, x.shape[0])
        # print(adj_old)
        adj_new_nonzero = adj_new.nonzero()
        graph_new = matrixtodict(adj_new_nonzero)

        edge_index_tensor_new = torch.tensor(edge_index_list_new)
        return addedgelist,removeedgelist,edge_index_tensor_new,graph_new,adj_new

    def gen_parameters(self,model,edge_index):
        model.eval()
        data = self.dataset[self.index]
        x=data.x
        # X = data.x.numpy()
        feature=model.get_feature(x)
        W1 = model.state_dict()['conv1.weight'].numpy()
        W2 = model.state_dict()['conv2.weight'].numpy()
        W = dict()
        W[0] = feature.detach().numpy()
        W[1] = W1
        W[2] = W2
        layernumbers = 2

        Hold = dict()
        Hold[0] = feature.detach().numpy()
        Hold[1], Hold[2], Hold[3] = model.back(feature, edge_index, edge_index)
        Hold[1] = Hold[1].detach().numpy()
        Hold[2] = Hold[2].detach().numpy()
        Hold[3] = Hold[3].detach().numpy()
        return W,Hold,feature
    def guding_edges(self,edge_index,graph_old,edges_dict_old,addedgelist, removeedgelist ):
        edge_index_list = edge_index.numpy().tolist()
        edge_index_list_new = copy.deepcopy(edge_index_list)
        # random.seed()
        removeedgeindex = []
        removeedgesave = []


        addedgesave = []
        data = self.dataset[self.index]
        x = data.x


        for path in removeedgelist:
            removeedgeindex.append(edges_dict_old[(path[0], path[1])])
            removeedgeindex.append(edges_dict_old[(path[1], path[0])])

        removeedgeindex = list(set(removeedgeindex))
        removeedgeindex = sorted(removeedgeindex)

        for j in reversed(removeedgeindex):
            # if [edge_index_list_new[0][j], edge_index_list_new[1][j]] in removeedgelist or [edge_index_list_new[1][j], edge_index_list_new[0][j]] in removeedgelist:
            #     print('yes')
            # else:
            #     print('false')
            # print((edge_index_list_new[0][j], edge_index_list_new[1][j]))
            del edge_index_list_new[0][j]
            del edge_index_list_new[1][j]
        # print('len edge',len(edge_index_list_new[0]))
        for addpath in addedgelist:
            edge_index_list_new[0].append(addpath[0])
            edge_index_list_new[1].append(addpath[1])
            edge_index_list_new[1].append(addpath[0])
            edge_index_list_new[0].append(addpath[1])
        adj_new = rumor_construct_adj_matrix(edge_index_list_new, x.shape[0])
        # print(adj_old)
        adj_new_nonzero = adj_new.nonzero()
        graph_new = matrixtodict(adj_new_nonzero)

        edge_index_tensor_new = torch.tensor(edge_index_list_new)
        return addedgelist, removeedgelist, edge_index_tensor_new, graph_new, adj_new
    def gen_case_edge(self):
        data = self.dataset[self.index]
        # print(data)
        #
        # print(len(data.edge_index[0]))

        x, edge_index = data.x, data.edge_index

        edge_index_list = data.edge_index.numpy().tolist()
        # for i in range(0, x.shape[0]):
        #     edge_index_list[0].append(i)
        #     edge_index_list[1].append(i)
        edge_index = torch.tensor(edge_index_list)
        adj_old = rumor_construct_adj_matrix(edge_index_list, x.shape[0])
        # print(adj_old)
        adj_old_nonzero = adj_old.nonzero()
        graph_old = matrixtodict(adj_old_nonzero)
        edges_dict_old = dict()
        for i, node in enumerate(edge_index_list[0]):
            edges_dict_old[(node, edge_index_list[1][i])] = i

        return x,edge_index,graph_old,edges_dict_old,adj_old





























