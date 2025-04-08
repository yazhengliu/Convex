from utils.utils_deeplift import *
from utils.evalation import metrics,metrics_graph,metrics_node
class deeplift():
    def __init__(self,Hold,Hnew,W,goal,addgoalpath,removegoalpath,feature,\
                 layernumbers,topk_pathlist,model,edges_new,edges_old,dataset):
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
        self.edges_old = edges_old
        self.dataset = dataset

    def select_importantpath(self, deepliftresultma,removeedgelist,addedgelist):
        old_index=np.argmax(self.Hold[2 * self.layernumbers - 1][self.goal])
        new_index=np.argmax(self.Hnew[2 * self.layernumbers - 1][self.goal])
        deepliftresult=dict()
        goal_logits_list=[]
        for j in range(0, len(self.removegoalpath)):
            strpath = []
            for pathindex in self.removegoalpath[j]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            # if old_index != new_index:
            #     deepliftresult[c] = deepliftresultma[j][new_index]-deepliftresultma[j][old_index]
            # else:
            #     deepliftresult[c]=deepliftresultma[j][new_index]
        for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
            strpath = []
            for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            # if old_index != new_index:
            #     deepliftresult[c] = deepliftresultma[j][new_index]-deepliftresultma[j][old_index]
            # else:
            #     deepliftresult[c]=deepliftresultma[j][new_index]

        sortdeeplift = sorted(deepliftresult.items(), key=lambda item: item[1], reverse=True)

        pa_add = []
        pa_remove = []
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(self.topk_pathlist)):
            print('l', l)
            topk_path = self.topk_pathlist[l]
            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortdeeplift[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in self.addgoalpath:
                    pa_add.append(deepliftpath)
                if deepliftpath in self.removegoalpath:
                    pa_remove.append(deepliftpath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask  = metrics_node(model, self.feature, self.goal, pa_add, pa_remove, edges_new,
                                           self.dataset,'mask',removeedgelist,addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add = metrics_node(model, self.feature, self.goal,pa_add,pa_remove, self.edges_old,self.dataset,'add',removeedgelist,addedgelist).cal()
            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list

    def select_importantpath_graph(self, deepliftresultma,removeedgelist,addedgelist):
        H_old=np.mean(self.Hold[2 * self.layernumbers - 1],axis=0)
        H_new=np.mean(self.Hnew[2 * self.layernumbers - 1],axis=0)

        old_index=np.argmax(H_old)
        new_index=np.argmax(H_new)
        deepliftresult=dict()
        goal_logits_list=[]
        for j in range(0, len(self.removegoalpath)):
            strpath = []
            for pathindex in self.removegoalpath[j]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            # deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index]-deepliftresultma[j][old_index]
            else:
                deepliftresult[c]=deepliftresultma[j][new_index]
        for j in range(len(self.removegoalpath), len(self.removegoalpath) + len(self.addgoalpath)):
            strpath = []
            for pathindex in self.addgoalpath[j - len(self.removegoalpath)]:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            # deepliftresult[c] = deepliftresultma[j][new_index] - deepliftresultma[j][old_index]
            if old_index != new_index:
                deepliftresult[c] = deepliftresultma[j][new_index]-deepliftresultma[j][old_index]
            else:
                deepliftresult[c]=deepliftresultma[j][new_index]

        sortdeeplift = sorted(deepliftresult.items(), key=lambda item: item[1], reverse=True)
        print('deeplift',sortdeeplift)

        pa_add = []
        pa_remove = []
        goal_logits_mask_list=[]
        goal_logits_add_list=[]
        for l in range(0, len(self.topk_pathlist)):
            print('l', l)
            topk_path = self.topk_pathlist[l]
            for i in range(0, topk_path):
                deepliftpath = []
                s1 = sortdeeplift[i][0].split(',')
                for j in s1:
                    deepliftpath.append(int(j))
                if deepliftpath in self.addgoalpath:
                    pa_add.append(deepliftpath)
                if deepliftpath in self.removegoalpath:
                    pa_remove.append(deepliftpath)
            edges_new = self.edges_new
            model = self.model
            goal_logits_mask =  metrics_graph(model, self.feature, pa_add, pa_remove, edges_new,
                                        self.Hnew[self.layernumbers * 2 - 1],'mask',removeedgelist,addedgelist).cal()
            goal_logits_mask_list.append(goal_logits_mask)
            goal_logits_add =metrics_graph(model, self.feature,  pa_add,pa_remove,self.edges_old,
                                        self.Hold[self.layernumbers * 2 - 1],'add',removeedgelist,addedgelist).cal()

            goal_logits_add_list.append(goal_logits_add)
            # print('mask',goal_logits_mask)
            # print('add', goal_logits_add)
        return goal_logits_mask_list, goal_logits_add_list





