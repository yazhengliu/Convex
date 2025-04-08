from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph,metrics_node
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
class gnnlrp():
    def __init__(self,Hold,Hnew,W,goal,addgoalpath,removegoalpath,feature,\
                 layernumbers,topk_pathlist,model,edges_new,graph_new,\
                 graph_old,edges_old,dataset,newgoalpaths,oldgoalpaths):
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
        self.graph_new=graph_new
        self.graph_old = graph_old
        self.edges_old = edges_old
        self.dataset=dataset
        self.newgoalpaths=newgoalpaths
        self.oldgoalpaths=oldgoalpaths

    def contribution_value(self):

        goal = self.goal
        Hold = self.Hold
        Hnew = self.Hnew
        W = self.W
        graph_new=self.graph_new
        graph_old=self.graph_old
        layernumbers = self.layernumbers
        oldgoalpaths=self.oldgoalpaths
        newgoalpaths=self.newgoalpaths

        path_XAI_dict = dict()
        path_XAI_old, _ = \
            XAIxiugai(goal, np.argmax(Hold[2 * layernumbers - 1][goal]), layernumbers,
                                     oldgoalpaths,
                                     H=Hold, W=W)
        path_XAI_new, _ = \
            XAIxiugai(goal, np.argmax(Hnew[2 * layernumbers - 1][goal]), layernumbers,
                                     newgoalpaths,
                                     H=Hnew, W=W)
        for key, value in path_XAI_new.items():
            if key in path_XAI_old.keys():
                path_XAI_dict[key] = value - path_XAI_old[key]
            if key not in path_XAI_old.keys():
                path_XAI_dict[key] = value
        for key, value in path_XAI_old.items():
            if key not in path_XAI_new.keys():
                path_XAI_dict[key] = value
        sort_gnnlrp = sorted(path_XAI_dict.items(), key=lambda item: item[1], reverse=True)
        return sort_gnnlrp
    def contribution_value_graph(self):

        goal = self.goal
        Hold = self.Hold
        Hnew = self.Hnew
        W = self.W
        graph_new=self.graph_new
        graph_old=self.graph_old
        layernumbers = self.layernumbers
        oldgoalpaths=self.oldgoalpaths
        newgoalpaths=self.newgoalpaths
        H_new = np.mean(self.Hnew[self.layernumbers * 2 - 1], axis=0)
        H_old = np.mean(self.Hold[self.layernumbers * 2 - 1], axis=0)

        path_XAI_dict = dict()
        pa_old = {}
        pa_new={}
        for path in oldgoalpaths:
            if path[0] not in pa_old.keys():
                pa_old[path[0]] = [path]
            else:
                pa_old[path[0]].append(path)
        for path in newgoalpaths:
            if path[0] not in pa_new.keys():
                pa_new[path[0]] = [path]
            else:
                pa_new[path[0]].append(path)
        path_XAI_old_zong={}
        for goal,oldpath,in pa_old.items():
            path_XAI_old, _ = \
                XAIxiugai(goal, np.argmax(H_old), layernumbers,
                          oldpath,
                          H=Hold, W=W)
            # print(path_XAI_old)
            path_XAI_old_zong=Merge(path_XAI_old_zong,path_XAI_old)
        # print(sum(path_XAI_old_zong.values())/self.feature.shape[0])
        path_XAI_new_zong = {}
        for goal, newpath, in pa_new.items():
            path_XAI_new, _ = \
                XAIxiugai(goal, np.argmax(H_new), layernumbers,
                          newpath,
                          H=Hnew, W=W)
            # print(path_XAI_old)
            path_XAI_new_zong = Merge(path_XAI_new_zong, path_XAI_new)
        # print(sum(path_XAI_new_zong.values())/self.feature.shape[0])

        path_XAI_dict={}
        for key, value in path_XAI_new_zong.items():
            if key in path_XAI_old_zong.keys():
                path_XAI_dict[key] = value - path_XAI_old_zong[key]
            if key not in path_XAI_old_zong.keys():
                path_XAI_dict[key] = value
        for key, value in path_XAI_old_zong.items():
            if key not in path_XAI_new_zong.keys():
                path_XAI_dict[key] = value
        sort_gnnlrp = sorted(path_XAI_dict.items(), key=lambda item: item[1], reverse=True)
        return sort_gnnlrp
    def select_importantpath(self,removeedgelist, addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_gnnlrp=self.contribution_value()
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnlrp[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)

            edges_new=self.edges_new
            model=self.model
            goal_logits_mask = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, edges_new,
                                            self.dataset, 'mask', removeedgelist, addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            # pa_add = []
            # pa_remove = []
            # for i in range(0, topk_path):
            #     lrppath = []
            #     s1 = sort_gnnlrp[i][0].split(',')
            #     for j in s1:
            #         lrppath.append(int(j))
            #     if lrppath in self.addgoalpath:
            #         pa_add.append(lrppath)
            #     if lrppath in self.removegoalpath:
            #         pa_remove.append(lrppath)
            goal_logits_add = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, self.edges_old,
                                           self.dataset, 'add', removeedgelist, addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list
    def select_importantpath_fortime(self,removeedgelist, addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_gnnlrp=self.contribution_value()
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnlrp[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)


    def select_importantpath_graph(self,removeedgelist,
                                            addedgelist):
        topk_pathlist=self.topk_pathlist
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        sort_gnnlrp=self.contribution_value_graph()
        print('gnnlrp',sort_gnnlrp)
        oldgoalpaths = self.oldgoalpaths
        newgoalpaths = self.newgoalpaths

        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnlrp[i][0].split(',')
                for j in s1:
                    lrppath.append(int(j))
                if lrppath in self.addgoalpath:
                    pa_add.append(lrppath)
                if lrppath in self.removegoalpath:
                    pa_remove.append(lrppath)
                if lrppath in oldgoalpaths and lrppath in newgoalpaths:
                    pa_add.append(lrppath)

            edges_new=self.edges_new
            model=self.model
            goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                             self.Hnew[self.layernumbers * 2 - 1], 'mask', removeedgelist,
                                             addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            pa_add = []
            pa_remove = []
            for i in range(0, topk_path):
                lrppath = []
                s1 = sort_gnnlrp[i][0].split(',')
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
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list



