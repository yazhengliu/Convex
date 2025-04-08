from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph,metrics_node
class linear():
    def __init__(self, Hold, Hnew, W, goal, addgoalpath, removegoalpath, feature, \
                 layernumbers, topk_pathlist, model, edges_new,nclass,edges_old,dataset):
        self.Hold = Hold
        self.Hnew = Hnew
        self.goal = goal
        self.W = W
        self.addgoalpath = addgoalpath
        self.removegoalpath = removegoalpath
        self.feature = feature
        self.layernumbers = layernumbers
        self.topk_pathlist = topk_pathlist
        self.model = model
        self.edges_new = edges_new
        self.nclass=nclass
        self.edges_old = edges_old
        self.dataset=dataset

    def select_importantpath(self,deepliftresultma,removeedgelist, addedgelist):
        topk_pathlist = self.topk_pathlist

        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]
            number3 = sum(
                self.Hnew[self.layernumbers * 2 - 1][self.goal] - self.Hold[self.layernumbers * 2 - 1][self.goal])
            linearlist = main_linear(number3, topk_path, self.goal, deepliftresultma, self.nclass, self.Hnew,
                                     self.layernumbers)
            linearresultdict = dict()
            for j in range(0, len(self.removegoalpath)):
                strpath = []
                for pathindex in self.removegoalpath[j]:
                    strpath.append(str(pathindex))
                c = ','.join(strpath)
                linearresultdict[c] = linearlist[j]
            for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
                strpath = []
                for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                    strpath.append(str(pathindex))
                c = ','.join(strpath)
                linearresultdict[c] = linearlist[j]
            sortlinearresultdict = sorted(linearresultdict.items(), key=lambda item: item[1], reverse=True)
            pa_add = []
            pa_remove = []

            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortlinearresultdict[i][0].split(',')
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
    def select_importantpath_graph(self,deepliftresultma,removeedgelist,
                                             addedgelist):
        topk_pathlist = self.topk_pathlist
        H_new = np.mean(self.Hnew[self.layernumbers * 2 - 1], axis=0)
        H_old = np.mean(self.Hold[self.layernumbers * 2 - 1], axis=0)
        number3 = sum(
           H_new - H_old)

        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(topk_pathlist)):
            print('l', l)
            topk_path = topk_pathlist[l]

            linearlist = main_linear_graph(number3, topk_path, deepliftresultma, self.nclass, H_new,
                                     self.layernumbers)
            linearresultdict = dict()
            for j in range(0, len(self.removegoalpath)):
                strpath = []
                for pathindex in self.removegoalpath[j]:
                    strpath.append(str(pathindex))
                c = ','.join(strpath)
                linearresultdict[c] = linearlist[j]
            for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
                strpath = []
                for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                    strpath.append(str(pathindex))
                c = ','.join(strpath)
                linearresultdict[c] = linearlist[j]
            sortlinearresultdict = sorted(linearresultdict.items(), key=lambda item: item[1], reverse=True)
            print('linear',sortlinearresultdict)

            pa_add = []
            pa_remove = []

            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortlinearresultdict[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in self.addgoalpath:
                    pa_add.append(deepliftpath)
                if deepliftpath in self.removegoalpath:
                    pa_remove.append(deepliftpath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask = metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                             self.Hnew[self.layernumbers * 2 - 1], 'mask', removeedgelist,
                                             addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_graph(model, self.feature, pa_add, pa_remove, self.edges_old,
                                            self.Hold[self.layernumbers * 2 - 1], 'add', removeedgelist,
                                            addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)

            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list


