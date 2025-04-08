from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph,metrics_node
class topk():
    def __init__(self,  goal,  addgoalpath, removegoalpath, feature, \
                 layernumbers, topk_pathlist, model, edges_new,edges_old,dataset):

        self.goal = goal
        self.addgoalpath = addgoalpath
        self.removegoalpath = removegoalpath
        self.feature = feature
        self.layernumbers = layernumbers
        self.topk_pathlist = topk_pathlist
        self.model = model
        self.edges_new = edges_new
        self.edges_old = edges_old
        self.dataset=dataset

    def select_importantpath(self,deepliftresultma,removeedgelist,addedgelist):
        topk_pathlist=self.topk_pathlist
        deepliftresult_topk = dict()  # AxiomPath-Topk
        for j in range(0, len(self.removegoalpath)):
            strpath = []
            for pathindex in self.removegoalpath[j]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult_topk[c] = sum(deepliftresultma[j])
        for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
            strpath = []
            for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult_topk[c] = sum(deepliftresultma[j])
        sort_topk = sorted(deepliftresult_topk.items(), key=lambda item: item[1], reverse=True)
        pa_add = []
        pa_remove = []
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sort_topk[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in self.addgoalpath:
                    pa_add.append(deepliftpath)
                if deepliftpath in self.removegoalpath:
                    pa_remove.append(deepliftpath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, edges_new,
                                            self.dataset, 'mask', removeedgelist, addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, self.edges_old,
                                           self.dataset, 'add', removeedgelist, addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list
    def select_importantpath_graph(self,deepliftresultma,Hold,Hnew,removeedgelist,
                                             addedgelist):
        topk_pathlist=self.topk_pathlist
        deepliftresult_topk = dict()  # AxiomPath-Topk
        for j in range(0, len(self.removegoalpath)):
            strpath = []
            for pathindex in self.removegoalpath[j]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult_topk[c] = sum(deepliftresultma[j])
        for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
            strpath = []
            for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult_topk[c] = sum(deepliftresultma[j])
        sort_topk = sorted(deepliftresult_topk.items(), key=lambda item: item[1], reverse=True)
        print('topk_k',sort_topk)

        pa_add = []
        pa_remove = []
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sort_topk[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in self.addgoalpath:
                    pa_add.append(deepliftpath)
                if deepliftpath in self.removegoalpath:
                    pa_remove.append(deepliftpath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                             Hnew[self.layernumbers * 2 - 1], 'mask', removeedgelist,
                                             addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_graph(model, self.feature, pa_add, pa_remove, self.edges_old,
                                           Hold[self.layernumbers * 2 - 1], 'add', removeedgelist,
                                            addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list


