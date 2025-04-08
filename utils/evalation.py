from utils.utils_deeplift import *
import scipy.stats
import copy
class metrics():
    def __init__(self,model,x,goal,pa_add,pa_remove,edges_new,dataset):
        self.model=model
        self.x=x
        self.goal=goal
        self.pa_add=pa_add
        self.pa_remove=pa_remove
        self.edges_new=edges_new
        self.dataset=dataset

    def cal(self):
        model_mask=self.model
        x=self.x
        goal=self.goal
        pa_remove=self.pa_remove
        pa_add=self.pa_add
        edges_new=self.edges_new
        edges_index_new = torch.tensor(edges_new)
        edges_dict_new = dict()
        for i, node in enumerate(edges_new[0]):
            edges_dict_new[(node, edges_new[1][i])] = i

        edges_index_1, edges_index_2, edges_index_3 = edge_index_both(edges_dict_new, pa_add,
                                                                                     pa_remove,
                                                                                     edges_new)

        if self.dataset=='Chi' or self.dataset=='NYC' or self.dataset=='Zip':
            if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                a = model_mask.forward(x, edges_index_1, edges_index_2)[goal].detach().numpy()
                if edges_index_3.tolist() != [[], []]:
                    b = model_mask.forward(x, edges_index_new, edges_index_3)[goal].detach().numpy()
                    return a+b
                else:
                    return a
        elif self.dataset=='pheme' or self.dataset=='weibo':
            if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                a = model_mask.back_gcn(x, edges_index_1, edges_index_2)[goal].detach().numpy()
                if edges_index_3.tolist() != [[], []]:
                    b = model_mask.back_gcn(x, edges_index_new, edges_index_3)[goal].detach().numpy()
                    return a + b
                else:
                    return a
class metrics_node():
    def __init__(self,model,x,goal,pa_add,pa_remove,edges_new,dataset,type,removeedgelist,addedgelist):
        self.model=model
        self.x=x
        self.goal=goal
        self.pa_add=pa_add
        self.pa_remove=pa_remove
        self.edges_new=edges_new
        self.dataset=dataset
        self.type=type
        self.removeedgelist=removeedgelist
        self.addedgelist=addedgelist

    def cal(self):
        model_mask=self.model
        x=self.x
        goal=self.goal
        pa_remove=self.pa_remove
        pa_add=self.pa_add
        edges_new=self.edges_new
        edges_index_new = torch.tensor(edges_new)
        edges_dict_new = dict()
        for i, node in enumerate(edges_new[0]):
            edges_dict_new[(node, edges_new[1][i])] = i
        if self.type=='add':
            edges_index_1, edges_index_2=edge_index_both_g0(edges_dict_new, pa_add,
                                                                                     pa_remove,
                                                                                     edges_new,self.removeedgelist)
        elif self.type=='mask':
            edges_index_1, edges_index_2 = edge_index_both_g1(edges_dict_new, pa_add,
                                                              pa_remove,
                                                              edges_new, self.addedgelist)


        if self.dataset=='Chi' or self.dataset=='NYC' or self.dataset=='Zip':
            if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                a = model_mask.forward(x, edges_index_1, edges_index_2)[goal].detach().numpy()
                return a
        elif self.dataset=='pheme' or self.dataset=='weibo':
            if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
                a = model_mask.back_gcn(x, edges_index_1, edges_index_2)[goal].detach().numpy()
                return a

class  metrics_graph():
    def __init__(self,model,x,pa_add,pa_remove,edges_new,H,type,removeedgelist,addedgelist):
        self.model=model
        self.x=x
        self.pa_add=pa_add
        self.pa_remove=pa_remove
        self.edges_new=edges_new
        self.H=H
        self.type=type
        self.removeedgelist=removeedgelist
        self.addedgelist = addedgelist



    def cal(self):
        model_mask=self.model
        x=self.x

        pa_remove=self.pa_remove
        pa_add=self.pa_add
        edges_new=self.edges_new
        # edges_index_new = torch.tensor(edges_new)
        edges_dict_new = dict()
        for i, node in enumerate(edges_new[0]):
            edges_dict_new[(node, edges_new[1][i])] = i

        pa_add_firstneighbor= {}
        for path in pa_add:
            if path[0] not in pa_add_firstneighbor.keys():
                pa_add_firstneighbor[path[0]]=[path]
            else:
                pa_add_firstneighbor[path[0]].append(path)
        pa_remove_firstneighbor = {}
        for path in pa_remove:
            if path[0] not in pa_remove_firstneighbor.keys():
                pa_remove_firstneighbor[path[0]] = [path]
            else:
                pa_remove_firstneighbor[path[0]].append(path)
        pa_zong=dict()
        for key,value in pa_add_firstneighbor.items():
            if key in pa_remove_firstneighbor.keys():
                pa_zong[key]=(value,pa_remove_firstneighbor[key])
            else:
                pa_zong[key]=(value,0)
        for key,value in pa_remove_firstneighbor.items():
            if key not in pa_add_firstneighbor.keys():
                pa_zong[key]=(0,value)
        # print('pa_zong',pa_zong)
        H_ceshi=np.array(self.H,copy=True)



        for goal,paths in pa_zong.items():
            addpath,removepath=paths[0],paths[1]
            if addpath==0:
                addpath=[]
            if removepath==0:
                removepath=[]
            # print(goal,addpath,removepath)
            # print(goal,addpath,removepath)
            if addpath!=[] or removepath!=[]:
                if self.type == 'add':
                    # print('len edges',len(edges_new[0]))
                    edges_index_1, edges_index_2 = edge_index_both_g0(edges_dict_new, addpath,
                                                                      removepath,
                                                                      edges_new, self.removeedgelist)
                    H_ceshi[goal] = model_mask.forward_pre(x, edges_index_1, edges_index_2).detach().numpy()[goal]
                elif self.type == 'mask':
                    edges_index_1, edges_index_2 = edge_index_both_g1(edges_dict_new, addpath,
                                                                      removepath,
                                                                      edges_new, self.addedgelist)
                    H_ceshi[goal] = model_mask.forward_pre(x, edges_index_1, edges_index_2).detach().numpy()[goal]


            # print('ceshi goal',H_ceshi[goal])
            # print('new goal',Hnew[3][goal])

            # if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
            #     # print(goal)
            #     a = model_mask.forward_pre(x, edges_index_1, edges_index_2).detach().numpy()
            #     # print(a)
            #     # a=np.mean(a,axis=0)
            #     if edges_index_3.tolist() != [[], []]:
            #         b = model_mask.forward_pre(x, edges_index_new, edges_index_3).detach().numpy()
            #         # b=np.mean(b,axis=0)
            #
            #         H_ceshi[goal]=a[goal]+b[goal]
            #         # print(goal, H[goal])
            #         # print('true goal',Hnew[3][goal])
            #         # return np.mean(a+b,axis=0)
            #     else:
            #         H_ceshi[goal]=a[goal]


        return np.mean(H_ceshi,axis=0)
class  metrics_graph_ceshi():
    def __init__(self,model,x,pa_add,pa_remove,edges_new,goal):
        self.model=model
        self.x=x
        self.pa_add=pa_add
        self.pa_remove=pa_remove
        self.edges_new=edges_new
        self.goal=goal

    def cal(self):
        model_mask=self.model
        x=self.x
        goal=self.goal

        pa_remove=self.pa_remove
        pa_add=self.pa_add
        edges_new=self.edges_new
        edges_index_new = torch.tensor(edges_new)
        edges_dict_new = dict()
        for i, node in enumerate(edges_new[0]):
            edges_dict_new[(node, edges_new[1][i])] = i

        edges_index_1, edges_index_2, edges_index_3 = edge_index_both(edges_dict_new, pa_add,
                                                                                     pa_remove,
                                                                                     edges_new)
        # print(edges_index_1)
        # print(edges_index_2)
        # print(edges_index_3)
        if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
            a = model_mask.forward_pre(x, edges_index_1, edges_index_2)[goal].detach().numpy()
            if edges_index_3.tolist() != [[], []]:
                b = model_mask.forward_pre(x, edges_index_new, edges_index_3)[goal].detach().numpy()
                return a + b
            else:
                return a



class metrics_link():
    def __init__(self,model,x,goal_1,goal_2,pa_add,pa_remove,edges_new,hidden,H,type,removeedgelist,addedgelist):
        self.model=model
        self.x=x
        self.goal_1=goal_1
        self.goal_2 = goal_2
        self.pa_add=pa_add
        self.pa_remove=pa_remove
        self.edges_new=edges_new
        self.hidden=hidden
        self.H=H
        self.type=type
        self.removeedgelist=removeedgelist
        self.addedgelist=addedgelist

    def cal(self):
        model_mask=self.model
        x=self.x
        goal_1=self.goal_1
        goal_2 = self.goal_2
        pa_remove=self.pa_remove
        pa_add=self.pa_add
        edges_new=self.edges_new
        edges_index_new = torch.tensor(edges_new)
        edges_dict_new = dict()
        for i, node in enumerate(edges_new[0]):
            edges_dict_new[(node, edges_new[1][i])] = i



        start_encode =torch.tensor(self.H[3][goal_1]).reshape(1, self.hidden)
        end_encode = torch.tensor(self.H[3][goal_2]).reshape(1, self.hidden)

        goal1_addpath=[]
        goal1_removepath = []
        for path in pa_add:
            if path[0]==goal_1:
                goal1_addpath.append(path)
        for path in pa_remove:
            if path[0]==goal_1:
                goal1_removepath.append(path)
        # print('goal1_addpath',goal1_addpath)
        # print(' goal1_removepath', goal1_removepath)

        goal2_addpath = []
        goal2_removepath = []
        for path in pa_add:
            if path[0] == goal_2:
                goal2_addpath.append(path)
        for path in pa_remove:
            if path[0] == goal_2:
                goal2_removepath.append(path)
        if self.type=='add':
            edges_index_1, edges_index_2= edge_index_both_g0(edges_dict_new, goal1_addpath,
                                                                          goal1_removepath,
                                                                          edges_new,self.removeedgelist)
        elif self.type=='mask':
            edges_index_1, edges_index_2= edge_index_both_g1(edges_dict_new, goal1_addpath,
                                                                             goal1_removepath,
                                                                             edges_new, self.addedgelist)
        # print(edges_index_1)
        # print(edges_index_2)
        # print(edges_index_3)
        a = model_mask.back_1(x.to(torch.float32), edges_index_1, edges_index_2)
        start_encode = (a)[goal_1].reshape(1, self.hidden)
        # if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
        #
        #    a = model_mask.back_1(x.to(torch.float32), edges_index_1, edges_index_2)
        #    if edges_index_3.tolist() != [[], []]:
        #             b = model_mask.back_1(x.to(torch.float32), edges_index_new, edges_index_3)
        #             start_encode = (a + b)[goal_1].reshape(1, self.hidden)
        #    else:
        #        start_encode = (a)[goal_1].reshape(1, self.hidden)
        if self.type == 'add':
            edges_index_1, edges_index_2 = edge_index_both_g0(edges_dict_new, goal2_addpath,
                                                              goal2_removepath,
                                                              edges_new, self.removeedgelist)
        elif self.type == 'mask':
            edges_index_1, edges_index_2 = edge_index_both_g1(edges_dict_new, goal2_addpath,
                                                              goal2_removepath,
                                                              edges_new, self.addedgelist)
        a = model_mask.back_1(x.to(torch.float32), edges_index_1, edges_index_2)
        end_encode = (a)[goal_2].reshape(1, self.hidden)

        # edges_index_1, edges_index_2, edges_index_3 = edge_index_both(edges_dict_new, goal2_addpath,
        #                                                                      goal2_removepath,
        #                                                                      edges_new)
        # if edges_index_1.tolist() != [[], []] and edges_index_2.tolist() != [[], []]:
        #
        #     a = model_mask.back_1(x.to(torch.float32), edges_index_1, edges_index_2)
        #     if edges_index_3.tolist() != [[], []]:
        #         b = model_mask.back_1(x.to(torch.float32), edges_index_new, edges_index_3)
        #         end_encode = (a + b)[goal_2].reshape(1, self.hidden)
        #     else:
        #         end_encode = (a)[goal_2].reshape(1, self.hidden)
        # print('start',start_encode.detach().numpy())
        # print('end',end_encode.detach().numpy())

        encode = torch.cat([start_encode, end_encode], dim=1)

        c = model_mask.back_2(encode).squeeze().detach().numpy()
        return c






class metrics_KL():
    def __init__(self,Hold,Hnew,goal,layer):
        self.goal=goal
        self.Hold=Hold
        self.Hnew=Hnew
        self.layer=layer
    def KL(self,add,mask):
        layer=self.layer
        mask_KL=[]
        add_KL=[]

        for i in range(len(mask)):
            if mask[i] is not None:
                mask_KL.append(KL_divergence(softmax(self.Hold[layer * 2 - 1][self.goal]), softmax(mask[i])))
            else:
                mask_KL.append(KL_divergence(softmax(self.Hold[layer * 2 - 1][self.goal]),softmax(self.Hnew[layer * 2 - 1][self.goal])))

            # print(self.Hold)
            # print(self.Hold[layer*2-1][self.goal])

        for i in range(len(add)):
            if add[i] is not None:
                add_KL.append(KL_divergence(softmax(self.Hnew[layer * 2 - 1][self.goal]),softmax(add[i])))
            else:
                add_KL.append(KL_divergence(softmax(self.Hnew[layer * 2 - 1][self.goal]), softmax(self.Hold[layer * 2 - 1][self.goal])))


        return mask_KL,add_KL
class metrics_KL_graph():
    def __init__(self,Hold,Hnew):

        self.Hold=Hold
        self.Hnew=Hnew

    def KL(self,add,mask):
        mask_KL=[]
        add_KL=[]

        for i in range(len(mask)):
            # print(self.Hold)
            # print(self.Hold[layer*2-1][self.goal])
            if mask[i] is not None:
                mask_KL.append(KL_divergence(softmax(self.Hold),softmax(mask[i])))
            else:
                mask_KL.append(KL_divergence(softmax(self.Hold),
                                             softmax(self.Hnew)))


        for i in range(len(add)):
            if add[i] is not None:
                add_KL.append(KL_divergence(softmax(self.Hnew),softmax(add[i])))
            else:
                add_KL.append(KL_divergence(softmax(self.Hnew),
                                            softmax(self.Hold)))

        return mask_KL,add_KL
class metrics_prob():
    def __init__(self,Hold,Hnew,goal,layer):
        self.goal=goal
        self.Hold=Hold
        self.Hnew=Hnew
        self.layer=layer
    def prob(self,add,mask):
        layer=self.layer
        mask_prob=[]
        add_prob=[]
        old_index = np.argmax(self.Hold[2 * layer - 1][self.goal])
        new_index = np.argmax(self.Hnew[2 * layer - 1][self.goal])
        for i in range(len(add)):
            if add[i] is not None:
                add_prob.append(abs(softmax(add[i])[new_index]-softmax(self.Hnew[layer*2-1][self.goal])[new_index]))
            else:
                add_prob.append(
                    abs(softmax(self.Hold[layer * 2 - 1][self.goal])[new_index] - softmax(self.Hnew[layer * 2 - 1][self.goal])[new_index]))

        for i in range(len(mask)):
            if mask[i] is not None:
                mask_prob.append(abs(softmax(mask[i])[old_index] - softmax(self.Hold[layer * 2 - 1][self.goal])[old_index]))
            else:
                mask_prob.append(
                    abs(softmax(self.Hnew[layer * 2 - 1][self.goal])[old_index] - softmax(self.Hold[layer * 2 - 1][self.goal])[old_index]))

        return mask_prob,add_prob

class metrics_prob_graph():
    def __init__(self,Hold,Hnew):

        self.Hold=Hold
        self.Hnew=Hnew

    def prob(self,add,mask):

        mask_prob=[]
        add_prob=[]
        old_index = np.argmax(self.Hold)
        new_index = np.argmax(self.Hnew)
        # print(old_index)
        # print(new_index)
        # print(self.Hnew)
        for i in range(len(add)):
            # print('add',add[i][0])
            if add[i] is not None:
                add_prob.append(abs(softmax(add[i])[new_index]-softmax(self.Hnew)[new_index]))
            else:
                add_prob.append(
                    abs(softmax(self.Hold)[new_index] -
                        softmax(self.Hnew)[new_index]))

        for i in range(len(mask)):
            if mask[i] is not None:
                mask_prob.append(abs(softmax(mask[i])[old_index] - softmax(self.Hold)[old_index]))
            else:
                mask_prob.append(
                    abs(softmax(self.Hnew)[old_index] -
                        softmax(self.Hold)[old_index]))

        return mask_prob,add_prob

