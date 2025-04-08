from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_link
class deeplift_link():
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

    def select_importantpath(self, deepliftresultma,Hold,Hnew,removeedgelist,addedgelist):

        addgoalpath_1 = self.addgoalpath_1
        addgoalpath_2 = self.addgoalpath_2
        removegoalpath_1 = self.removegoalpath_1
        removegoalpath_2 = self.removegoalpath_2
        index_new = self.index_new
        index_old = self.index_old

        old_index=np.argmax(np.argmax(self.Hold_mlp[1][index_old]))
        new_index=np.argmax(np.argmax(self.Hnew_mlp[1][index_new]))
        deepliftresult=dict()

        for j in range(0, len(addgoalpath_2)):
            strpath = []
            for pathindex in addgoalpath_2[j]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            #deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            else:
                deepliftresult[c] = deepliftresultma[j][new_index]

        for j in range(len(addgoalpath_2), len(addgoalpath_2) + len(addgoalpath_1)):
            strpath = []
            for pathindex in addgoalpath_1[j - len(addgoalpath_2)]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            #deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            else:
                deepliftresult[c] = deepliftresultma[j][new_index]
        for j in range(len(addgoalpath_2) + len(addgoalpath_1),
                       len(removegoalpath_2) + len(addgoalpath_2) + len(addgoalpath_1)):
            strpath = []
            for pathindex in removegoalpath_2[j - (len(addgoalpath_2) + len(addgoalpath_1))]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            # deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            else:
                deepliftresult[c] = deepliftresultma[j][new_index]

        for j in range(len(removegoalpath_2) + len(addgoalpath_2) + len(addgoalpath_1),
                       len(removegoalpath_2) + len(addgoalpath_2) + len(addgoalpath_1) + len(removegoalpath_1)):
            strpath = []
            for pathindex in removegoalpath_1[
                j - (len(addgoalpath_2) + len(addgoalpath_1) + len(removegoalpath_2))]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            # deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            else:
                deepliftresult[c] = deepliftresultma[j][new_index]
        sortdeeplift = sorted(deepliftresult.items(), key=lambda item: item[1], reverse=True)


        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(self.topk_pathlist)):
            print('l', l)
            topk_path = self.topk_pathlist[l]
            pa_add = []
            pa_remove = []

            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortdeeplift[i][0].split(',')
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
                                            self.hidden,Hnew,'mask',removeedgelist,addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_link(model, feature, self.goal_1, self.goal_2,  pa_add,pa_remove, self.edges_old,
                                           self.hidden,Hold,'add',removeedgelist,addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list





