import numpy as np
from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph
from utils.load_data import *
from torch_geometric.nn import global_mean_pool
class grad():
    def __init__(self,Hold,Hnew,W,goal,addgoalpath,removegoalpath,feature,\
                 layernumbers,topk_pathlist,model,edges_new,edges_old,graph_new,graph_old,dataset,newgoalpaths,oldgoalpaths):
        self.Hold=Hold
        self.Hnew=Hnew
        self.goal=goal
        self.W=W
        self.addgoalpath=addgoalpath
        self.removegoalpath=removegoalpath
        self.feature=feature
        self.layernumbers=layernumbers
        self.topk_pathlist=topk_pathlist
        self.model=model
        self.edges_new=edges_new
        self.edges_old=edges_old
        self.graph_new = graph_new
        self.graph_old = graph_old
        self.dataset=dataset
        self.newgoalpaths = newgoalpaths
        self.oldgoalpaths = oldgoalpaths

    def forward_tensor(self,adj, layernumbers, W):  # 有relu
        hiddenmatrix = dict()
        # adj = torch.tensor(adj, requires_grad=True)
        # adj=sparse_mx_to_torch_sparse_tensor(adj)
        relu = torch.nn.ReLU(inplace=False)
        hiddenmatrix[0] = W[0]

        h = torch.sparse.mm(adj, W[0])

        hiddenmatrix[1] = torch.mm(h, W[1])
        hiddenmatrix[2] = relu(hiddenmatrix[1])
        # hiddenmatrix[1].retain_grad()
        for i in range(1, layernumbers):
            h = torch.sparse.mm(adj, hiddenmatrix[2 * i])
            hiddenmatrix[2 * i + 1] = torch.mm(h, W[i + 1])
            if i != layernumbers - 1:
                hiddenmatrix[2 * i + 2] = relu(hiddenmatrix[2 * i + 1])
            # hiddenmatrix[i + 1].retain_grad()
        return hiddenmatrix

    def forward_tensor_graph(self,adj, layernumbers, W):  # 有relu
        hiddenmatrix = dict()
        # adj = torch.tensor(adj, requires_grad=True)
        # adj=sparse_mx_to_torch_sparse_tensor(adj)
        relu = torch.nn.ReLU(inplace=False)
        hiddenmatrix[0] = W[0]

        h = torch.mm(adj, W[0])

        hiddenmatrix[1] = torch.mm(h, W[1])
        hiddenmatrix[2] = relu(hiddenmatrix[1])
        # hiddenmatrix[1].retain_grad()
        for i in range(1, layernumbers):
            h = torch.mm(adj, hiddenmatrix[2 * i])
            hiddenmatrix[2 * i + 1] = torch.mm(h, W[i + 1])
            if i != layernumbers - 1:
                hiddenmatrix[2 * i + 2] = relu(hiddenmatrix[2 * i + 1])
            # hiddenmatrix[i + 1].retain_grad()
        return hiddenmatrix
    def mapsubgoalpaths(self,goalpaths, submapping):
        mapsub = []
        for path in goalpaths:
            subpath = []
            for node in path:
                subpath.append(submapping[node])
            mapsub.append(subpath)
        return mapsub

    def grad(self,H_tensor, pathlist, goal, k, adj_tensor,layernumbers):
        re = dict()

        # H_tensor=H_tensor.cuda()
        H_tensor[layernumbers * 2 - 1][goal][k].backward(retain_graph=True)
        adj_grad = adj_tensor.grad
        # print('adj_grad',adj_grad)
        # adj_grad=adj_grad.numpy()
        # print('adj_grad', adj_grad)
        for path in pathlist:
            grad_attr = 0
            for i in range(0, len(path) - 1):
                grad_attr = grad_attr + adj_grad[path[i]][path[i + 1]].item()
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] = grad_attr
        return re
    def grad_graph(self,H_tensor, pathlist, x, k, adj_tensor,layernumbers):
        re = dict()

        # H_tensor=H_tensor.cuda()
        batch = torch.tensor([0] * x.shape[0])
        print(global_mean_pool(H_tensor[layernumbers * 2 - 1], batch))
        global_mean_pool(H_tensor[layernumbers * 2 - 1], batch)[0][k].backward(retain_graph=True)

        # np.mean(H_tensor[layernumbers * 2 - 1],axis=0)[k].backward(retain_graph=True)
        adj_grad = adj_tensor.grad
        # print('adj_grad',adj_grad)
        # adj_grad=adj_grad.numpy()
        # print('adj_grad', adj_grad)
        for path in pathlist:
            grad_attr = 0
            for i in range(0, len(path) - 1):
                grad_attr = grad_attr + adj_grad[path[i]][path[i + 1]].item()
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] = grad_attr
        return re


    def logits(self):
        node_list = []
        goal=self.goal
        layernumbers=self.layernumbers
        graph_new = self.graph_new
        graph_old = self.graph_old
        edges_new_tensor = torch.tensor(self.edges_new)
        edges_old_tensor = torch.tensor(self.edges_old)
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        W_tensor = dict()

        # W_tensor[0] = W_tensor[0].double()
        W_tensor[1] = torch.tensor(self.W[1], requires_grad=False)
        # W_tensor[1] = W_tensor[1].double()
        W_tensor[2] = torch.tensor(self.W[2], requires_grad=False)

        subset_new, edge_new_sub, _, _ = k_hop_subgraph(
            goal, layernumbers, edges_new_tensor, relabel_nodes=False,
            num_nodes=None)
        subset_old, edge_old_sub, _, _ = k_hop_subgraph(
            goal, layernumbers, edges_old_tensor, relabel_nodes=False,
            num_nodes=None)
        # print('edge_new_sub',edge_new_sub)
        # print('subset_new',subset_new)
        for node in subset_new:
            if node.item() not in node_list:
                node_list.append(node.item())
        for node in subset_old:
            if node.item() not in node_list:
                node_list.append(node.item())
        node_map = dict()
        for idx, node in enumerate(node_list):
            node_map[node] = idx
        node_map_reverse = {value: key for key, value in node_map.items()}
        mapedgesnew, subfetures_new, map_adj_new, sub_edges_dict = map_edges(edge_new_sub.tolist(), node_map,
                                                                             self.feature.detach().numpy(), node_list)
        mapedgesold, subfetures_old, map_adj_old, sub_edges_dict_old = map_edges(edge_old_sub.tolist(), node_map,
                                                                                 self.feature.detach().numpy(),
                                                                                 node_list)

        #         submapping_new, subadj_new = utils_deeplift.subadj_map(subset_new, edge_new_sub)
        #         # print('submapping',submapping)
        #         # print('subadj',subadj)
        #         submapping_reverse_new = {value: key for key, value in submapping_new.items()}
        #         sub_features = utils_deeplift.subH(subset_new, submapping_new, Hold, layernumbers)[0]
        W_tensor[0] = torch.tensor(subfetures_new, requires_grad=False)
        W_tensor[0] = W_tensor[0].to(torch.float32)

        subadj_new = sparse_mx_to_torch_sparse_tensor(map_adj_new)
        subadj_new = torch.tensor(subadj_new, requires_grad=True)
        H_tensornew = self.forward_tensor(subadj_new, layernumbers, W_tensor)
        sub_newpaths = self.mapsubgoalpaths(newgoalpaths, node_map)

        gradresult_new = self.grad(H_tensornew, sub_newpaths, node_map[goal],
                              np.argmax(self.Hnew[2 * layernumbers - 1][goal]), subadj_new,self.layernumbers)

        #         submapping_old, subadj_old = utils_deeplift.subadj_map(subset_old, edge_old_sub)
        #         submapping_reverse_old = {value: key for key, value in submapping_old.items()}
        #
        subadj_old = sparse_mx_to_torch_sparse_tensor(map_adj_old)
        subadj_old = torch.tensor(subadj_old, requires_grad=True)

        W_tensor[0] = torch.tensor(subfetures_old, requires_grad=False)
        W_tensor[0] = W_tensor[0].to(torch.float32)
        H_tensorold = self.forward_tensor(subadj_old, layernumbers, W_tensor)
        sub_oldpaths = self.mapsubgoalpaths(oldgoalpaths, node_map)

        gradresult_old = self.grad(H_tensorold, sub_oldpaths, node_map[goal],
                              np.argmax(self.Hold[2 * layernumbers - 1][goal]), subadj_old,self.layernumbers)
        #         gradresult_old_original = pathresult(gradresult_old, submapping_reverse_old)
        #         gradresult_new_original = pathresult(gradresult_new, submapping_reverse_new)
        #         # print('gradresult_old_original',gradresult_old_original)
        #         sortgrad_original = dict()
        sortgrad = dict()
        for key, value in gradresult_new.items():
            if key in gradresult_old.keys():
                sortgrad[key] = value - gradresult_old[key]
            if key not in gradresult_old.keys():
                sortgrad[key] = value
        for key, value in gradresult_old.items():
            if key not in gradresult_new.keys():
                sortgrad[key] = value
        # sortgrad = mappathresult(sortgrad_original, submapping_new)
        sort_grad = sorted(sortgrad.items(), key=lambda item: item[1], reverse=True)

        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list = []

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]

            if self.dataset=='Chi' or self.dataset=='NYC' or self.dataset=='Zip':
                pa_add = []
                pa_remove = []
                for i in range(0, topk_path):
                    lrppath = []
                    s1 = sort_grad[i][0].split(',')
                    lrppath_original = []
                    for j in s1:
                        lrppath.append(int(j))
                        lrppath_original.append(node_map_reverse[int(j)])
                    if lrppath_original in self.addgoalpath:
                        pa_add.append(lrppath)
                    if lrppath_original in self.removegoalpath:
                        pa_remove.append(lrppath)
                    if lrppath_original in oldgoalpaths and lrppath_original in newgoalpaths:
                        pa_add.append(lrppath)
                edges_index_1, edges_index_2, edges_index_3 = edge_index_both(sub_edges_dict, pa_add,
                                                                              pa_remove, mapedgesnew)
                if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                    a = \
                    self.model.forward(torch.tensor(subfetures_new).to(torch.float32), edges_index_1, edges_index_2)[
                        node_map[goal]].detach().numpy()
                    if edges_index_3.tolist() != [[], []]:
                        b = \
                            self.model.forward(torch.tensor(subfetures_new).to(torch.float32),
                                               torch.tensor(mapedgesnew),
                                               edges_index_3)[
                                node_map[goal]].detach().numpy()
                        goal_logits_mask_list.append(a + b)
                    else:
                        goal_logits_mask_list.append(a)
                else:
                    goal_logits_mask_list.append(None)
                pa_add = []
                pa_remove = []
                for i in range(0, topk_path):
                    lrppath = []
                    s1 = sort_grad[i][0].split(',')
                    lrppath_original = []
                    for j in s1:
                        lrppath.append(int(j))
                        lrppath_original.append(node_map_reverse[int(j)])
                    if lrppath_original in self.addgoalpath:
                        pa_add.append(lrppath)
                    if lrppath_original in self.removegoalpath:
                        pa_remove.append(lrppath)



                edges_index_1, edges_index_2, edges_index_3 = edge_index_both(sub_edges_dict_old, pa_remove,
                                                                              pa_add, mapedgesold)
                if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                    a = \
                    self.model.forward(torch.tensor(subfetures_old).to(torch.float32), edges_index_1, edges_index_2)[
                        node_map[goal]].detach().numpy()
                    if edges_index_3.tolist() != [[], []]:
                        b = \
                            self.model.forward(torch.tensor(subfetures_old).to(torch.float32),
                                               torch.tensor(mapedgesold),
                                               edges_index_3)[
                                node_map[goal]].detach().numpy()
                        goal_logits_add_list.append(a + b)
                    else:
                        goal_logits_add_list.append(a)
                else:
                    goal_logits_add_list.append(None)
            elif self.dataset=='pheme' or self.dataset=='weibo':
                pa_add = []
                pa_remove = []
                for i in range(0, topk_path):
                    lrppath = []
                    s1 = sort_grad[i][0].split(',')
                    lrppath_original = []
                    for j in s1:
                        lrppath.append(int(j))
                        lrppath_original.append(node_map_reverse[int(j)])
                    if lrppath_original in self.addgoalpath:
                        pa_add.append(lrppath)
                    if lrppath_original in self.removegoalpath:
                        pa_remove.append(lrppath)
                    if lrppath_original in oldgoalpaths and lrppath_original in newgoalpaths:
                        pa_add.append(lrppath)
                edges_index_1, edges_index_2, edges_index_3 = edge_index_both(sub_edges_dict, pa_add,
                                                                              pa_remove, mapedgesnew)
                if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                    a = \
                        self.model.back_gcn(torch.tensor(subfetures_new).to(torch.float32), edges_index_1,
                                           edges_index_2)[
                            node_map[goal]].detach().numpy()
                    if edges_index_3.tolist() != [[], []]:
                        b = \
                            self.model.back_gcn(torch.tensor(subfetures_new).to(torch.float32),
                                               torch.tensor(mapedgesnew),
                                               edges_index_3)[
                                node_map[goal]].detach().numpy()
                        goal_logits_mask_list.append(a + b)
                    else:
                        goal_logits_mask_list.append(a)
                else:
                    goal_logits_mask_list.append(None)
                pa_add = []
                pa_remove = []
                for i in range(0, topk_path):
                    lrppath = []
                    s1 = sort_grad[i][0].split(',')
                    lrppath_original = []
                    for j in s1:
                        lrppath.append(int(j))
                        lrppath_original.append(node_map_reverse[int(j)])
                    if lrppath_original in self.addgoalpath:
                        pa_add.append(lrppath)
                    if lrppath_original in self.removegoalpath:
                        pa_remove.append(lrppath)

                edges_index_1, edges_index_2, edges_index_3 = edge_index_both(sub_edges_dict_old, pa_remove,
                                                                              pa_add, mapedgesold)
                if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                    a = \
                        self.model.back_gcn(torch.tensor(subfetures_old).to(torch.float32), edges_index_1,
                                           edges_index_2)[
                            node_map[goal]].detach().numpy()
                    if edges_index_3.tolist() != [[], []]:
                        b = \
                            self.model.back_gcn(torch.tensor(subfetures_old).to(torch.float32),
                                               torch.tensor(mapedgesold),
                                               edges_index_3)[
                                node_map[goal]].detach().numpy()
                        goal_logits_add_list.append(a + b)
                    else:
                        goal_logits_add_list.append(a)
                else:
                    goal_logits_add_list.append(None)




        return goal_logits_mask_list,goal_logits_add_list
    def logits_fortime(self):
        node_list = []
        goal=self.goal
        layernumbers=self.layernumbers
        graph_new = self.graph_new
        graph_old = self.graph_old
        edges_new_tensor = torch.tensor(self.edges_new)
        edges_old_tensor = torch.tensor(self.edges_old)
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        W_tensor = dict()

        # W_tensor[0] = W_tensor[0].double()
        W_tensor[1] = torch.tensor(self.W[1], requires_grad=False)
        # W_tensor[1] = W_tensor[1].double()
        W_tensor[2] = torch.tensor(self.W[2], requires_grad=False)

        subset_new, edge_new_sub, _, _ = k_hop_subgraph(
            goal, layernumbers, edges_new_tensor, relabel_nodes=False,
            num_nodes=None)
        subset_old, edge_old_sub, _, _ = k_hop_subgraph(
            goal, layernumbers, edges_old_tensor, relabel_nodes=False,
            num_nodes=None)
        # print('edge_new_sub',edge_new_sub)
        # print('subset_new',subset_new)
        for node in subset_new:
            if node.item() not in node_list:
                node_list.append(node.item())
        for node in subset_old:
            if node.item() not in node_list:
                node_list.append(node.item())
        node_map = dict()
        for idx, node in enumerate(node_list):
            node_map[node] = idx
        node_map_reverse = {value: key for key, value in node_map.items()}
        mapedgesnew, subfetures_new, map_adj_new, sub_edges_dict = map_edges(edge_new_sub.tolist(), node_map,
                                                                             self.feature.detach().numpy(), node_list)
        mapedgesold, subfetures_old, map_adj_old, sub_edges_dict_old = map_edges(edge_old_sub.tolist(), node_map,
                                                                                 self.feature.detach().numpy(),
                                                                                 node_list)

        #         submapping_new, subadj_new = utils_deeplift.subadj_map(subset_new, edge_new_sub)
        #         # print('submapping',submapping)
        #         # print('subadj',subadj)
        #         submapping_reverse_new = {value: key for key, value in submapping_new.items()}
        #         sub_features = utils_deeplift.subH(subset_new, submapping_new, Hold, layernumbers)[0]
        W_tensor[0] = torch.tensor(subfetures_new, requires_grad=False)
        W_tensor[0] = W_tensor[0].to(torch.float32)

        subadj_new = sparse_mx_to_torch_sparse_tensor(map_adj_new)
        subadj_new = torch.tensor(subadj_new, requires_grad=True)
        H_tensornew = self.forward_tensor(subadj_new, layernumbers, W_tensor)
        sub_newpaths = self.mapsubgoalpaths(newgoalpaths, node_map)

        gradresult_new = self.grad(H_tensornew, sub_newpaths, node_map[goal],
                              np.argmax(self.Hnew[2 * layernumbers - 1][goal]), subadj_new,self.layernumbers)

        #         submapping_old, subadj_old = utils_deeplift.subadj_map(subset_old, edge_old_sub)
        #         submapping_reverse_old = {value: key for key, value in submapping_old.items()}
        #
        subadj_old = sparse_mx_to_torch_sparse_tensor(map_adj_old)
        subadj_old = torch.tensor(subadj_old, requires_grad=True)

        W_tensor[0] = torch.tensor(subfetures_old, requires_grad=False)
        W_tensor[0] = W_tensor[0].to(torch.float32)
        H_tensorold = self.forward_tensor(subadj_old, layernumbers, W_tensor)
        sub_oldpaths = self.mapsubgoalpaths(oldgoalpaths, node_map)

        gradresult_old = self.grad(H_tensorold, sub_oldpaths, node_map[goal],
                              np.argmax(self.Hold[2 * layernumbers - 1][goal]), subadj_old,self.layernumbers)
        #         gradresult_old_original = pathresult(gradresult_old, submapping_reverse_old)
        #         gradresult_new_original = pathresult(gradresult_new, submapping_reverse_new)
        #         # print('gradresult_old_original',gradresult_old_original)
        #         sortgrad_original = dict()
        sortgrad = dict()
        for key, value in gradresult_new.items():
            if key in gradresult_old.keys():
                sortgrad[key] = value - gradresult_old[key]
            if key not in gradresult_old.keys():
                sortgrad[key] = value
        for key, value in gradresult_old.items():
            if key not in gradresult_new.keys():
                sortgrad[key] = value
        # sortgrad = mappathresult(sortgrad_original, submapping_new)
        sort_grad = sorted(sortgrad.items(), key=lambda item: item[1], reverse=True)



    def logits_graph(self,adj_old,adj_new,removeedgelist,
                                             addedgelist):
        node_list = []
        goal=self.goal
        layernumbers=self.layernumbers
        graph_new = self.graph_new
        graph_old = self.graph_old
        edges_new_tensor = torch.tensor(self.edges_new)
        edges_old_tensor = torch.tensor(self.edges_old)
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths
        H_new = np.mean(self.Hnew[self.layernumbers * 2 - 1], axis=0)
        H_old = np.mean(self.Hold[self.layernumbers * 2 - 1], axis=0)

        W_tensor = dict()

        # W_tensor[0] = W_tensor[0].double()
        W_tensor[1] = torch.tensor(self.W[1], requires_grad=False)
        # W_tensor[1] = W_tensor[1].double()
        W_tensor[2] = torch.tensor(self.W[2], requires_grad=False)


        W_tensor[0] = torch.tensor(self.feature, requires_grad=False)
        W_tensor[0] = W_tensor[0].to(torch.float32)

        adj_old_tensor = torch.tensor(adj_old.todense(), requires_grad=True)
        adj_new_tensor = torch.tensor(adj_new.todense(), requires_grad=True)

        H_tensorold = self.forward_tensor_graph(adj_old_tensor, layernumbers, W_tensor)
        gradresult_old = self.grad_graph(H_tensorold, oldgoalpaths, self.feature,
                              np.argmax(H_old), adj_old_tensor,self.layernumbers)
        H_tensornew = self.forward_tensor_graph(adj_new_tensor, layernumbers, W_tensor)
        gradresult_new = self.grad_graph(H_tensornew, newgoalpaths, self.feature,
                              np.argmax(H_new), adj_new_tensor,self.layernumbers)




        sortgrad = dict()
        for key, value in gradresult_new.items():
            if key in gradresult_old.keys():
                sortgrad[key] = value - gradresult_old[key]
            if key not in gradresult_old.keys():
                sortgrad[key] = value
        for key, value in gradresult_old.items():
            if key not in gradresult_new.keys():
                sortgrad[key] = value
        # sortgrad = mappathresult(sortgrad_original, submapping_new)
        sort_grad = sorted(sortgrad.items(), key=lambda item: item[1], reverse=True)
        print('grad',sort_grad)

        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list = []

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_grad[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))

                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                             self.Hnew[self.layernumbers * 2 - 1], 'mask', removeedgelist,
                                             addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_grad[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)
            goal_logits_add = metrics_graph(model, self.feature, pa_add, pa_remove, self.edges_old,
                                            self.Hold[self.layernumbers * 2 - 1], 'add', removeedgelist,
                                            addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)




        return goal_logits_mask_list,goal_logits_add_list



