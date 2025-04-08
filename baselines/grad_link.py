from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_link
from utils.utils_link import *

class grad_link():
    def __init__(self,Hold,Hnew,W,Hnew_mlp, Hold_mlp,W_mlp,goal_1,goal_2,index_new,index_old,\
                 layernumbers,hidden,dataset,graph_new,graph_old,addgoalpath_1,addgoalpath_2\
                 ,removegoalpath_1,removegoalpath_2, oldgoalpaths_1,oldgoalpaths_2,newgoalpaths_1,\
                 newgoalpaths_2,topk_pathlist,edges_new,edges_old,model,adj_old,adj_new):
        self.Hold=Hold
        self.Hnew=Hnew
        self.goal_1=goal_1
        self.goal_2 = goal_2
        self.Hnew_mlp=Hnew_mlp
        self.Hold_mlp=Hold_mlp
        self.layernumbers=layernumbers
        self.index_new=index_new
        self.index_old=index_old
        self.hidden=hidden
        self.dataset=dataset
        self.W=W
        self.W_mlp=W_mlp
        self.graph_new=graph_new
        self.graph_old=graph_old
        self.addgoalpath_1=addgoalpath_1
        self.addgoalpath_2=addgoalpath_2
        self.removegoalpath_1=removegoalpath_1
        self.removegoalpath_2=removegoalpath_2
        self.oldgoalpaths_1=oldgoalpaths_1
        self.oldgoalpaths_2=oldgoalpaths_2
        self.newgoalpaths_1=newgoalpaths_1
        self.newgoalpaths_2=newgoalpaths_2
        self.topk_pathlist=topk_pathlist
        self.edges_new = edges_new
        self.edges_old = edges_old
        self.model=model
        self.adj_old=adj_old
        self.adj_new=adj_new


    def contribution_value(self):
        goal_1=self.goal_1
        goal_2 = self.goal_2
        layernumbers=self.layernumbers
        Hold = self.Hold
        Hnew = self.Hnew
        W = self.W
        feature=self.dataset[0].x
        hidden=self.hidden
        newgoalpaths_1=self.newgoalpaths_1
        newgoalpaths_2=self.newgoalpaths_2
        oldgoalpaths_1=self.oldgoalpaths_1
        oldgoalpaths_2=self.oldgoalpaths_2
        adj_old=self.adj_old
        adj_new=self.adj_new

        adj_old_tensor = torch.tensor(adj_old.todense(), requires_grad=True)
        adj_new_tensor = torch.tensor(adj_new.todense(), requires_grad=True)
        W_tensor = dict()
        W_tensor[0] = torch.tensor(feature.to(torch.float32), requires_grad=False)
        # W_tensor[0] = W_tensor[0].double()
        W_tensor[1] = torch.tensor(W[1], requires_grad=False)
        # W_tensor[1] = W_tensor[1].double()
        W_tensor[2] = torch.tensor(W[2], requires_grad=False)
        # W_tensor[2] = W_tensor[2].double()
        H_tensorold = forward_tensor_link(adj_old_tensor, layernumbers, W_tensor)
        H_tensornew = forward_tensor_link(adj_new_tensor, layernumbers, W_tensor)

        path_grad_old_1 = grad(H_tensorold, oldgoalpaths_1, goal_1,
                               np.argmax(Hold[2 * layernumbers - 1][goal_1]), adj_old_tensor,layernumbers)
        path_grad_old_2 = grad(H_tensorold, oldgoalpaths_2, goal_2,
                               np.argmax(Hold[2 * layernumbers - 1][goal_2]), adj_old_tensor,layernumbers)
        path_grad_new_1 =grad(H_tensornew, newgoalpaths_1, goal_1,
                               np.argmax(Hnew[2 * layernumbers - 1][goal_1]), adj_new_tensor,layernumbers)
        path_grad_new_2 = grad(H_tensornew, newgoalpaths_2, goal_2,
                               np.argmax(Hnew[2 * layernumbers - 1][goal_2]), adj_new_tensor,layernumbers)

        path_grad_old = dict()
        for key, value in path_grad_old_1.items():
            if key in path_grad_old_2.keys():
                path_grad_old[key] = value + path_grad_old_2[key]
            if key not in path_grad_old_2.keys():
                path_grad_old[key] = value
        for key, value in path_grad_old_2.items():
            if key not in path_grad_old_1.keys():
                path_grad_old[key] = value

        path_grad_new = dict()
        for key, value in path_grad_new_1.items():
            if key in path_grad_new_2.keys():
                path_grad_new[key] = value + path_grad_new_2[key]
            if key not in path_grad_new_2.keys():
                path_grad_new[key] = value
        for key, value in path_grad_new_2.items():
            if key not in path_grad_new_1.keys():
                path_grad_new[key] = value

        path_grad_dict = dict()
        for key, value in path_grad_new.items():
            if key in path_grad_old.keys():
                path_grad_dict[key] = value - path_grad_old[key]
            if key not in path_grad_old.keys():
                path_grad_dict[key] = value
        for key, value in path_grad_old.items():
            if key not in path_grad_new.keys():
                path_grad_dict[key] = value
        sort_grad = sorted(path_grad_dict.items(), key=lambda item: item[1], reverse=True)
        return sort_grad


    def select_importantpath(self,removeedgelist, addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_grad=self.contribution_value()
        addgoalpath_1=self.addgoalpath_1
        addgoalpath_2=self.addgoalpath_2
        removegoalpath_1=self.removegoalpath_1
        removegoalpath_2=self.removegoalpath_2

        newgoalpaths_1 = self.newgoalpaths_1
        newgoalpaths_2 = self.newgoalpaths_2
        oldgoalpaths_1 = self.oldgoalpaths_1
        oldgoalpaths_2 = self.oldgoalpaths_2
        # print(sort_gnnlrp)


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
                if lrppath in addgoalpath_1 or lrppath in addgoalpath_2:
                    pa_add.append(lrppath)
                if lrppath in removegoalpath_1 or lrppath in removegoalpath_2:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths_1 and lrppath in newgoalpaths_1:
                    pa_add.append(lrppath)
                if lrppath in oldgoalpaths_2 and lrppath in newgoalpaths_2:
                    pa_add.append(lrppath)

            edges_new=self.edges_new
            model=self.model
            feature = self.dataset[0].x
            goal_logits_mask = metrics_link(model, feature, self.goal_1, self.goal_2, pa_add, pa_remove, edges_new,
                                            self.hidden, self.Hnew, 'mask', removeedgelist, addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_link(model, feature, self.goal_1, self.goal_2,  pa_add, pa_remove,self.edges_old,
                                           self.hidden, self.Hold, 'add', removeedgelist, addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # pa_add = []
            # pa_remove = []
            #
            # for i in range(0, topk_path):
            #     lrppath = []
            #     s1 = sort_grad[i][0].split(',')
            #     for j in s1:
            #         lrppath.append(int(j))
            #     if lrppath in addgoalpath_1 or lrppath in addgoalpath_2:
            #         pa_add.append(lrppath)
            #     if lrppath in removegoalpath_1 or lrppath in removegoalpath_2:
            #         pa_remove.append(lrppath)
            #
            # goal_logits_add = metrics_link(model, feature,  self.goal_1,self.goal_2, pa_remove, pa_add, self.edges_old,self.hidden,self.Hnew).cal()
            # goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list
    def select_importantpath_fortime(self,removeedgelist, addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_grad=self.contribution_value()
        addgoalpath_1=self.addgoalpath_1
        addgoalpath_2=self.addgoalpath_2
        removegoalpath_1=self.removegoalpath_1
        removegoalpath_2=self.removegoalpath_2

        newgoalpaths_1 = self.newgoalpaths_1
        newgoalpaths_2 = self.newgoalpaths_2
        oldgoalpaths_1 = self.oldgoalpaths_1
        oldgoalpaths_2 = self.oldgoalpaths_2
        # print(sort_gnnlrp)


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
                if lrppath in addgoalpath_1 or lrppath in addgoalpath_2:
                    pa_add.append(lrppath)
                if lrppath in removegoalpath_1 or lrppath in removegoalpath_2:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths_1 and lrppath in newgoalpaths_1:
                    pa_add.append(lrppath)
                if lrppath in oldgoalpaths_2 and lrppath in newgoalpaths_2:
                    pa_add.append(lrppath)






