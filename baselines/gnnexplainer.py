from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph,metrics_node
from torch_geometric.nn import  GNNExplainer
class gnnexplainer():
    def __init__(self,model,goal,feature,edges_new,edges_old,graph_new,graph_old,layernumbers,\
                 model_mask, topk_pathlist,addgoalpath,removegoalpath,dataset):
        self.model_mask = model_mask
        self.model=model
        self.goal=goal
        self.feature=feature
        self.edges_new = edges_new
        self.edges_old = edges_old
        self.graph_new = graph_new
        self.graph_old = graph_old
        self.layernumbers=layernumbers
        self.topk_pathlist = topk_pathlist
        self.addgoalpath = addgoalpath
        self.removegoalpath = removegoalpath
        self.dataset=dataset
    def edge_path(self,edge_mask,edges):
        node_weights=dict()
        for i in range(0, len(edge_mask)):
            # print(i)
            if edge_mask[i] != 0:
                # print(edge_mask1_list[i])
                # print(edge_index1[0][i])
                # print(edge_index1[1][i])

                node_weights[(edges[0][i], edges[1][i])] = edge_mask[i]
        return node_weights
    def path_value(self,weights,goalpaths):
        re = dict()
        for path in goalpaths:
            attr = 0
            for i in range(0, len(path) - 1):
                attr = attr + weights[(path[i],path[i+1])]
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] = attr
        return re

    def contribution_value(self):
        explainer = GNNExplainer(self.model, epochs=20, return_type='raw')
        newgoalpaths = dfs2(self.goal, self.goal, self.graph_new, self.layernumbers + 1, [], [])
        oldgoalpaths = dfs2(self.goal, self.goal, self.graph_old, self.layernumbers + 1, [], [])
        self.model.eval()
        # print(edge_index)
        # print(type(edge_index))

        _, edge_mask_old = explainer.explain_node(self.goal, self.feature, torch.tensor(self.edges_old))
        edge_mask_old_list = edge_mask_old.numpy().tolist()
        node_weights_old=self.edge_path(edge_mask_old_list,self.edges_old)
        # print('node_weights_old',node_weights_old)
        path_old=self.path_value(node_weights_old,oldgoalpaths)
        print('path_old',path_old)


        _, edge_mask_new = explainer.explain_node(self.goal, self.feature, torch.tensor(self.edges_new))
        edge_mask_new_list = edge_mask_new.numpy().tolist()
        node_weights_new = self.edge_path(edge_mask_new_list, self.edges_new)
        path_new=self.path_value(node_weights_new,newgoalpaths)
        print('path_new', path_new)
        #print('node_weights_new',node_weights_new)
        path_zong=dict()
        for key, value in path_new.items():
            if key in path_old.keys():
                path_zong[key] = value - path_old[key]
            if key not in path_old.keys():
                path_zong[key] = value
        for key, value in path_old.items():
            if key not in path_new.keys():
                path_zong[key] = value
        sort_gnnexplainer = sorted(path_zong.items(), key=lambda item: item[1], reverse=True)
        return  sort_gnnexplainer,newgoalpaths,oldgoalpaths

    def select_importantpath(self,removeedgelist,addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_gnnexplainer,newgoalpaths,oldgoalpaths=self.contribution_value()

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 =sort_gnnexplainer[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)


            edges_new=self.edges_new
            model=self.model_mask
            goal_logits_mask = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, edges_new,
                                            self.dataset, 'mask', removeedgelist, addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)

            # pa_add = []
            # pa_remove = []
            # for i in range(0, topk_path):
            #     lrppath = []
            #     s1 = sort_gnnexplainer[i][0].split(',')
            #     for j in s1:
            #         lrppath.append(int(j))
            #     if lrppath in self.addgoalpath:
            #         pa_add.append(lrppath)
            #     if lrppath in self.removegoalpath:
            #         pa_remove.append(lrppath)
            goal_logits_add = metrics_node(model, self.feature, self.goal,pa_add,pa_remove, self.edges_old,self.dataset,'add',removeedgelist,addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list
    def select_importantpath_fortime(self,removeedgelist,addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_gnnexplainer,newgoalpaths,oldgoalpaths=self.contribution_value()

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 =sort_gnnexplainer[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)



class gnnexplainer_graph():
    def __init__(self, dataset,model,  edges_new, edges_old, graph_new, graph_old, layernumbers, \
                  topk_pathlist, addgoalpath, removegoalpath,newgoalpaths,oldgoalpaths, data,index,Hold,Hnew,feature):

        self.model = model
        self.edges_new = edges_new
        self.edges_old = edges_old
        self.graph_new = graph_new
        self.graph_old = graph_old
        self.layernumbers = layernumbers
        self.topk_pathlist = topk_pathlist
        self.addgoalpath = addgoalpath
        self.removegoalpath = removegoalpath
        self.newgoalpaths=newgoalpaths
        self.oldgoalpaths=oldgoalpaths
        self.dataset = dataset
        self.data=data
        self.index=index
        self.Hold=Hold
        self.Hnew=Hnew
        self.feature=feature


    def edge_path(self, edge_mask, edges):
        node_weights = dict()
        for i in range(0, len(edge_mask)):
            # print(i)
            if edge_mask[i] != 0:
                # print(edge_mask1_list[i])
                # print(edge_index1[0][i])
                # print(edge_index1[1][i])

                node_weights[(edges[0][i], edges[1][i])] = edge_mask[i]
        return node_weights

    def path_value(self, weights, goalpaths):
        re = dict()
        for path in goalpaths:
            attr = 0
            for i in range(0, len(path) - 1):
                attr = attr + weights[(path[i], path[i + 1])]
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] = attr
        return re

    def contribution_value(self):
        oldgoalpaths=self.oldgoalpaths
        newgoalpaths=self.newgoalpaths
        explainer = GNNExplainer(self.model, epochs=20, return_type='raw')
        feature=self.data[self.index].x

        self.model.eval()
        # print(edge_index)
        # print(type(edge_index))

        edge_mask_old, _ = explainer.explain_graph(feature, torch.tensor(self.edges_old))
        edge_mask_old_list = edge_mask_old.numpy().tolist()
        node_weights_old = self.edge_path(edge_mask_old_list, self.edges_old)
        # print('node_weights_old',node_weights_old)
        path_old = self.path_value(node_weights_old, oldgoalpaths)
        # print('path_old', path_old)

        edge_mask_new,_ = explainer.explain_graph(feature, torch.tensor(self.edges_new))
        edge_mask_new_list = edge_mask_new.numpy().tolist()
        node_weights_new = self.edge_path(edge_mask_new_list, self.edges_new)
        path_new = self.path_value(node_weights_new, newgoalpaths)
        # print('path_new', path_new)
        # print('node_weights_new',node_weights_new)
        path_zong = dict()
        for key, value in path_new.items():
            if key in path_old.keys():
                path_zong[key] = value - path_old[key]
            if key not in path_old.keys():
                path_zong[key] = value
        for key, value in path_old.items():
            if key not in path_new.keys():
                path_zong[key] = -value
        sort_gnnexplainer = sorted(path_zong.items(), key=lambda item: item[1], reverse=True)
        return sort_gnnexplainer

    def select_importantpath(self,removeedgelist,
                                             addedgelist):
        topk_pathlist = self.topk_pathlist
        goal_logits_mask_list = []
        goal_logits_add_list = []
        sort_gnnexplainer= self.contribution_value()
        print('gnnexplainer',sort_gnnexplainer)
        feature = self.feature
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnexplainer[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)
            # print('pa_add',pa_add)
            # print('pa_remove',pa_remove)

            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_graph(model, feature, pa_add, pa_remove, edges_new,
                                             self.Hnew[self.layernumbers * 2 - 1], 'mask', removeedgelist,
                                             addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)

            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnexplainer[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)
            goal_logits_add = metrics_graph(model,feature, pa_add, pa_remove, self.edges_old,
                                            self.Hold[self.layernumbers * 2 - 1], 'add', removeedgelist,
                                            addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list


