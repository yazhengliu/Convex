from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_link
from utils.utils_link import main_linear_linear
class linear_link():
    def __init__(self, Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, index_new, index_old, \
                 layernumbers, hidden, dataset, graph_new, graph_old, addgoalpath_1, addgoalpath_2 \
                 , removegoalpath_1, removegoalpath_2, topk_pathlist, edges_new, edges_old, model, addedgelist,
                 removeedgelist):
        self.Hold = Hold
        self.Hnew = Hnew
        self.goal_1 = goal_1
        self.goal_2 = goal_2
        self.Hnew_mlp = Hnew_mlp
        self.Hold_mlp = Hold_mlp
        self.layernumbers = layernumbers
        self.index_new = index_new
        self.index_old = index_old
        self.hidden = hidden
        self.dataset = dataset
        self.W = W
        self.W_mlp = W_mlp
        self.graph_new = graph_new
        self.graph_old = graph_old
        self.addgoalpath_1 = addgoalpath_1
        self.addgoalpath_2 = addgoalpath_2
        self.removegoalpath_1 = removegoalpath_1
        self.removegoalpath_2 = removegoalpath_2
        self.topk_pathlist = topk_pathlist
        self.edges_new = edges_new
        self.edges_old = edges_old
        self.model = model
        self.addedgelist = addedgelist
        self.removeedgelist = removeedgelist

    def select_importantpath(self, deepliftresultma,Hold,Hnew, removeedgelist, addedgelist):

        addgoalpath_1 = self.addgoalpath_1
        addgoalpath_2 = self.addgoalpath_2
        removegoalpath_1 = self.removegoalpath_1
        removegoalpath_2 = self.removegoalpath_2
        index_new = self.index_new
        index_old = self.index_old
        Hold_mlp = self.Hold_mlp
        Hnew_mlp = self.Hnew_mlp

        old_index=np.argmax(np.argmax(self.Hold_mlp[1][index_old]))
        new_index=np.argmax(np.argmax(self.Hnew_mlp[1][index_new]))
        deepliftresult=dict()
        number3 = sum(Hnew_mlp[1][index_new] - Hold_mlp[1][index_old])
        layernumbers=self.layernumbers


        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        nclass=2
        for l in range(0, len(self.topk_pathlist)):
            print('l', l)
            topk_path = self.topk_pathlist[l]
            linearlist = main_linear_linear(number3, topk_path, index_new, deepliftresultma, nclass, Hnew_mlp,
                                                  layernumbers)
            linearresultdict = dict()
            for i in range(0, len(linearlist)):
                if i < len(addgoalpath_2):
                    strpath = []
                    for pathindex in addgoalpath_2[i]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    linearresultdict[c] = linearlist[i]
                if i >= len(addgoalpath_2) and i < len(addgoalpath_2) + len(addgoalpath_1):
                    strpath = []
                    for pathindex in addgoalpath_1[i - len(addgoalpath_2)]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    linearresultdict[c] = linearlist[i]
                if i >= len(addgoalpath_2) + len(addgoalpath_1) and i < len(addgoalpath_2) + len(
                        removegoalpath_2) + len(addgoalpath_1):
                    strpath = []
                    for pathindex in removegoalpath_2[i - len(addgoalpath_2) - len(addgoalpath_1)]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    linearresultdict[c] = linearlist[i]
                if i >= len(addgoalpath_2) + len(removegoalpath_2) + len(addgoalpath_1):
                    strpath = []
                    for pathindex in removegoalpath_1[
                        i - len(addgoalpath_2) - len(addgoalpath_1) - len(removegoalpath_2)]:
                        strpath.append(str(pathindex))
                    c = ','.join(strpath)
                    linearresultdict[c] = linearlist[i]
            sortlinearresultdict = sorted(linearresultdict.items(), key=lambda item: item[1], reverse=True)
            pa_add = []
            pa_remove = []

            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortlinearresultdict[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in addgoalpath_1 or deepliftpath in addgoalpath_2:
                    pa_add.append(deepliftpath)
                if deepliftpath in removegoalpath_1 or deepliftpath in removegoalpath_2:
                    pa_remove.append(deepliftpath)


            edges_new = self.edges_new
            model = self.model
            feature = self.dataset[0].x
            goal_logits_mask = metrics_link(model, feature, self.goal_1, self.goal_2, pa_add, pa_remove, edges_new,
                                            self.hidden, Hnew, 'mask', removeedgelist, addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_link(model, feature, self.goal_1, self.goal_2,  pa_add, pa_remove,self.edges_old,
                                           self.hidden, Hold, 'add', removeedgelist, addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list





