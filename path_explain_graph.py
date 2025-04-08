from utils.split_data import gen_Yelp_data
# dynamic_data=gen_dynamic_data('Chi',0,84,90,'month')
# adj_old,adj_new,edges_old,edges_new,graph_old,graph_new,edgelist,_,_=dynamic_data.gen_adj()
# Hold,Hnew,W,model_mask=dynamic_data.gen_parameters()
# print(edgelist)
from  utils.utils_deeplift import findnewpath,dfs2,deepliftrelumultigaijin,KL_divergence,softmax
import argparse
import json
import os
from utils.constant import time_step,path_number
from baselines.convex import convex
from baselines.gnnlrp import gnnlrp
from baselines.grad import grad
from baselines.gnnexplainer import gnnexplainer
from baselines.linear import linear
from baselines.topk import topk
from baselines.deeplift import deeplift
from utils.evalation import metrics_KL,metrics_prob
import numpy as np
from torch_geometric.datasets import TUDataset
import torch
from train_GCN_graph import GCN
from utils.split_data import gen_graph_data
from utils.evalation import metrics_KL_graph,metrics_prob_graph,metrics_graph,metrics_graph_ceshi
from utils.evalation import metrics
from torch_geometric.nn import  GNNExplainer
from baselines.gnnexplainer import gnnexplainer_graph
import matplotlib.pyplot as plt
from utils.utils_deeplift import sparse_mx_to_torch_sparse_tensor
import copy
import pickle


import random

def relu(x):
    s = np.where(x < 0, 0, x)
    return s
def forward_tensor(adj,layernumbers,W): #有relu
    hiddenmatrix = dict()
    # adj = torch.tensor(adj, requires_grad=True)
    # adj=sparse_mx_to_torch_sparse_tensor(adj)
    hiddenmatrix[0] = W[0]


    h = np.dot(adj, W[0])

    hiddenmatrix[1] = np.dot(h, W[1])
    hiddenmatrix[2]=relu(hiddenmatrix[1])
    # hiddenmatrix[1].retain_grad()
    for i in range(1, layernumbers):
        h = np.dot(adj, hiddenmatrix[2*i])
        hiddenmatrix[2*i + 1] = np.dot(h, W[i + 1])
        if i!=layernumbers-1:
            hiddenmatrix[2*i+2]=relu(hiddenmatrix[2*i + 1])
        # hiddenmatrix[i + 1].retain_grad()
    return hiddenmatrix

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutag')
    parser.add_argument('--type', type=str, default='both')
    args = parser.parse_args()
    if args.type=='both':
        addedgenum=5
        removeedgenum=5
    elif args.type=='add':
        addedgenum=5
        removeedgenum = 0
    elif args.type=='remove':
        addedgenum=0
        removeedgenum =5
    dataset = TUDataset('data/TUDataset', name='MUTAG', use_node_attr='True')
    # print(torch.load('data/' + 'TUDataset/' + 'GCN_model' + '.pth'))

    model = GCN(nfeat=7,hidden_channels=16,nclass=2)
    model.eval()
    model.load_state_dict(torch.load('data/' + 'TUDataset/' + 'GCN_model' + '.pth'))


    total_node=150
    total_node_list=random.sample(list(range(180)),total_node)
    random_num = 5


    Convex_add_prob = np.ones((total_node*random_num, 5))
    Convex_mask_prob = np.ones((total_node*random_num, 5))
    Convex_add_KL = np.ones((total_node*random_num, 5))
    Convex_mask_KL = np.ones((total_node*random_num, 5))

    DEEPLIFT_add_prob = np.ones((total_node*random_num, 5))
    DEEPLIFT_mask_prob = np.ones((total_node*random_num, 5))
    DEEPLIFT_add_KL = np.ones((total_node*random_num, 5))
    DEEPLIFT_mask_KL = np.ones((total_node*random_num, 5))

    Topk_add_prob= np.ones((total_node*random_num, 5))
    Topk_mask_prob = np.ones((total_node*random_num, 5))
    Topk_add_KL= np.ones((total_node*random_num, 5))
    Topk_mask_KL = np.ones((total_node*random_num, 5))

    Linear_add_prob = np.ones((total_node*random_num, 5))
    Linear_mask_prob = np.ones((total_node*random_num, 5))
    Linear_add_KL= np.ones((total_node*random_num, 5))
    Linear_mask_KL = np.ones((total_node*random_num, 5))

    GRAD_add_prob = np.ones((total_node*random_num, 5))
    GRAD_mask_prob = np.ones((total_node*random_num, 5))
    GRAD_add_KL = np.ones((total_node*random_num, 5))
    GRAD_mask_KL = np.ones((total_node*random_num, 5))

    GNNLRP_add_prob = np.ones((total_node*random_num, 5))
    GNNLRP_mask_prob = np.ones((total_node*random_num, 5))
    GNNLRP_add_KL = np.ones((total_node*random_num, 5))
    GNNLRP_mask_KL = np.ones((total_node*random_num, 5))

    graph_KL_list = []
    old_prob= np.ones((total_node*random_num, 1))
    new_prob = np.ones((total_node*random_num, 1))
    save_edge=dict()
    index_save=0




    flag=0
    for index in range(len(total_node_list)):
        # print('node index',index)
        node_index=total_node_list[index]
        # print('node number',dataset[node_index].x.shape[0])
        old_data = gen_graph_data(dataset, node_index, addedgenum, removeedgenum)
        x, edge_index_old, graph_old, edges_dict_old,adj_old = old_data.gen_original_edge()

        # print('total_edge',len(edge_index_old[0]))

        # print('edge_index_old',edge_index_old)
        # print(len(edge_index_old))
        # print('edge_index_old_list',edge_index_old.numpy().tolist())
        layernumbers=2

        print(adj_old)
        # for i in range(len(edge_index_old[0])):
        #     if edge_index_old[0][i]==16:
        #         print(edge_index_old[i])
        W, Hold,feature = old_data.gen_parameters(model, edge_index_old)
        Hold_ceshi = forward_tensor(adj_old.todense(), layernumbers, W)


        for idx in range(random_num):
            print('idx', idx)
            addedgelist, removeedgelist, edge_index_new, graph_new, \
            adj_new = old_data.random_edges(edge_index_old, graph_old, edges_dict_old)
            _, Hnew,_ = old_data.gen_parameters(model, edge_index_new)
            # print('addedgelist',addedgelist)
            # print('removeedgelist',removeedgelist)
            # print('graph_old',graph_old)
            # print('graph_new',graph_new)
            # print(edges_dict_old)
            graph_KL=KL_divergence(softmax(np.mean(Hnew[layernumbers*2-1],axis=0)),softmax(np.mean(Hold[layernumbers*2-1],axis=0)))
            graph_KL_list.append(graph_KL)

            old_prob[index_save][0]=softmax(np.mean(Hold[layernumbers*2-1],axis=0))[np.argmax(softmax(np.mean(Hold[layernumbers*2-1],axis=0)))]
            new_prob[index_save][0]=softmax(np.mean(Hnew[layernumbers*2-1],axis=0))[np.argmax(softmax(np.mean(Hnew[layernumbers*2-1],axis=0)))]
            # print('old prob',old_prob)


            print('graph_KL',graph_KL)
            if graph_KL>0.01:
                index_save_edge=dict()
                index_save_edge['addedgelist']=addedgelist
                index_save_edge['removeedgelist']=removeedgelist
                save_edge[str(index_save)]=index_save_edge
                edges_dict_new = dict()
                for i, node in enumerate(edge_index_new[0]):
                    edges_dict_new[(node.item(), edge_index_new[1][i].item())] = i
                # print(edges_dict_new)
                # for key,value in edges_dict_old.items():
                #     if key not in edges_dict_new.keys():
                #         print('remove',key)
                # for key, value in edges_dict_new.items():
                #     if key not in edges_dict_old.keys():
                #         print('add', key)

                layernumbers = 2
                addgoalpath = []
                removegoalpath = []
                newgoalpaths = []
                oldgoalpaths = []
                # print(np.mean(Hold[layernumbers * 2 - 1], axis=0))
                KL_eval = metrics_KL_graph(
                    np.mean(Hold[layernumbers * 2 - 1], axis=0), np.mean(Hnew[layernumbers * 2 - 1], axis=0))
                prob_eval = metrics_prob_graph(np.mean(Hold[layernumbers * 2 - 1], axis=0),
                                               np.mean(Hnew[layernumbers * 2 - 1], axis=0))


                select_pathlist = path_number(args.dataset, args.type, len(addgoalpath) + len(removegoalpath))

                addgoalpath = []
                removegoalpath = []
                for goal in range(0, x.shape[0]):
                    # addgoalpath = []
                    # removegoalpath = []
                    addgoalpath = addgoalpath + findnewpath(addedgelist, graph_new, layernumbers, goal)

                    removegoalpath = removegoalpath + findnewpath(removeedgelist, graph_old, layernumbers, goal)
                    newgoalpaths = newgoalpaths + dfs2(goal, goal, graph_new, layernumbers + 1, [], [])
                    oldgoalpaths = oldgoalpaths + dfs2(goal, goal, graph_old, layernumbers + 1, [], [])
                print('addgoalpath', len(addgoalpath))
                print('removegoalpath', len(removegoalpath))




                select_pathlist = path_number(args.dataset, args.type, len(addgoalpath) + len(removegoalpath))
                select_pathlist=[select_pathlist[0]]

                convex_method = convex(Hold, Hnew, W, goal, addedgelist, removeedgelist, addgoalpath, removegoalpath, \
                                       feature, layernumbers, 2, select_pathlist, \
                                       model, edge_index_new.numpy().tolist(), edge_index_old.numpy().tolist(),
                                       args.dataset,
                                       args.type)

                contriution_value = convex_method.contribution_value()
                # print(contriution_value)
                contriution_value = contriution_value / dataset[total_node_list[index]].x.shape[0]

                true_abs = np.mean(Hnew[layernumbers * 2 - 1], axis=0) - np.mean(Hold[layernumbers * 2 - 1], axis=0)
                pred_abs = sum(contriution_value)
                print('true', true_abs)
                print('pred', pred_abs)
                # print(contriution_value)

                convex_logists_mask, convex_logists_add = convex_method.select_importantpath_graph(contriution_value)
                # print('pred old',convex_logists_mask[0])
                # print('old',np.mean(Hold[layernumbers * 2 - 1], axis=0))
                # # print('pred new',convex_logists_add[0])
                # print('new',np.mean(Hnew[layernumbers * 2 - 1], axis=0))
                #
                convex_mask_KL, convex_add_KL = KL_eval.KL(convex_logists_add, convex_logists_mask)
                convex_mask_prob, convex_add_prob = prob_eval.prob(convex_logists_add, convex_logists_mask)

                Convex_mask_KL[index_save] = np.array(convex_mask_KL)
                Convex_add_KL[index_save] = np.array(convex_add_KL)
                Convex_mask_prob[index_save] = np.array(convex_mask_prob)
                Convex_add_prob[index_save] = np.array(convex_add_prob)
                print('ceshi index',index_save)
                print('convex_mask_KL', convex_mask_KL)
                print('convex_add_KL', convex_add_KL)
                print('convex_mask_prob', convex_mask_prob)
                print('convex_add_prob', convex_add_prob)

                edges_new = edge_index_new.numpy().tolist()
                edges_old = edge_index_old.numpy().tolist()

                deeplift_method = deeplift(Hold, Hnew, W, goal, addgoalpath, removegoalpath, feature, \
                                           layernumbers, select_pathlist, model, edges_new, edges_old, args.dataset)
                deeplift_logists_mask, deeplift_logists_add = deeplift_method.select_importantpath_graph(
                    contriution_value, removeedgelist, addedgelist)
                deeplift_mask_KL, deeplift_add_KL = KL_eval.KL(deeplift_logists_add, deeplift_logists_mask)
                deeplift_mask_prob, deeplift_add_prob = prob_eval.prob(deeplift_logists_add, deeplift_logists_mask)

                DEEPLIFT_mask_KL[index_save] = np.array(deeplift_mask_KL)
                DEEPLIFT_add_KL[index_save] = np.array(deeplift_add_KL)
                DEEPLIFT_mask_prob[index_save] = np.array(deeplift_mask_prob)
                DEEPLIFT_add_prob[index_save] = np.array(deeplift_add_prob)

                print('deeplift_mask_KL', deeplift_mask_KL)
                print('deeplift_add_KL', deeplift_add_KL)
                print('deeplift_mask_prob', deeplift_mask_prob)
                print('deeplift_add_prob', deeplift_add_prob)

                topk_method = topk(goal, addgoalpath, removegoalpath, feature, layernumbers, select_pathlist,
                                   model, edges_new, edges_old, args.dataset)
                topk_logists_mask, topk_logists_add = topk_method.select_importantpath_graph(contriution_value, Hold,
                                                                                             Hnew, removeedgelist,
                                                                                             addedgelist)
                topk_mask_KL, topk_add_KL = KL_eval.KL(topk_logists_add, topk_logists_mask)
                topk_mask_prob, topk_add_prob = prob_eval.prob(topk_logists_add, topk_logists_mask)

                Topk_mask_KL[index_save] = np.array(topk_mask_KL)
                Topk_add_KL[index_save] = np.array(topk_add_KL)
                Topk_mask_prob[index_save] = np.array(topk_mask_prob)
                Topk_add_prob[index_save] = np.array(topk_add_prob)

                print('topk_mask_KL', topk_mask_KL)
                print('topk_add_KL', topk_add_KL)
                print('topk_mask_prob', topk_mask_prob)
                print('topk_add_prob', topk_add_prob)

                linear_method = linear(Hold, Hnew, W, goal, addgoalpath, removegoalpath, feature, layernumbers,
                                       select_pathlist, \
                                       model, edges_new, 2, edges_old, args.dataset)
                linear_logists_mask, linear_logists_add = linear_method.select_importantpath_graph(contriution_value,
                                                                                                   removeedgelist,
                                                                                                   addedgelist)

                linear_mask_KL, linear_add_KL = KL_eval.KL(linear_logists_add, linear_logists_mask)
                linear_mask_prob, linear_add_prob = prob_eval.prob(linear_logists_add, linear_logists_mask)

                Linear_mask_KL[index_save] = np.array(linear_mask_KL)
                Linear_add_KL[index_save] = np.array(linear_add_KL)
                Linear_mask_prob[index_save] = np.array(linear_mask_prob)
                Linear_add_prob[index_save] = np.array(linear_add_prob)

                print('linear_mask_KL', linear_mask_KL)
                print('linear_add_KL', linear_add_KL)
                print('linear_mask_prob', linear_mask_prob)
                print('linear_add_prob', linear_add_prob)

                gnnlrp_method = gnnlrp(Hold, Hnew, W, goal, addgoalpath, removegoalpath, feature, layernumbers \
                                       , select_pathlist, model, edges_new, graph_new, graph_old, edges_old,
                                       args.dataset, newgoalpaths, oldgoalpaths)

                gnnlrp_logists_mask, gnnlrp_logists_add = gnnlrp_method.select_importantpath_graph(removeedgelist,
                                                                                                   addedgelist)
                gnnlrp_mask_KL, gnnlrp_add_KL = KL_eval.KL(gnnlrp_logists_add, gnnlrp_logists_mask)
                gnnlrp_mask_prob, gnnlrp_add_prob = prob_eval.prob(gnnlrp_logists_add, gnnlrp_logists_mask)

                if len(gnnlrp_mask_KL) == len(select_pathlist):
                    GNNLRP_mask_KL[index_save] = np.array(gnnlrp_mask_KL)
                    GNNLRP_mask_prob[index_save] = np.array(gnnlrp_mask_prob)

                if len(gnnlrp_add_KL) == len(select_pathlist):
                    GNNLRP_add_KL[index_save] = np.array(gnnlrp_add_KL)
                    GNNLRP_add_prob[index_save] = np.array(gnnlrp_add_prob)
                print('gnnlrp_mask_KL', gnnlrp_mask_KL)
                print('gnnlrp_add_KL', gnnlrp_add_KL)
                print('gnnlrp_mask_prob', gnnlrp_mask_prob)
                print('gnnlrp_add_prob', gnnlrp_add_prob)

                grad_method = grad(Hold, Hnew, W, goal, addgoalpath, removegoalpath, feature, layernumbers, \
                                   select_pathlist, model, edges_new, edges_old, graph_new, graph_old, args.dataset, \
                                   newgoalpaths, oldgoalpaths)
                grad_logits_mask, grad_logits_add = grad_method.logits_graph(adj_old, adj_new, removeedgelist,
                                                                             addedgelist)
                # print(grad_logits_add)
                grad_mask_KL, grad_add_KL = KL_eval.KL(grad_logits_add, grad_logits_mask)
                grad_mask_prob, grad_add_prob = prob_eval.prob(grad_logits_add, grad_logits_mask)

                if len(grad_mask_KL) == len(select_pathlist):
                    GRAD_mask_KL[index_save] = np.array(grad_mask_KL)
                    GRAD_mask_prob[index_save] = np.array(grad_mask_prob)
                if len(grad_add_prob) == len(select_pathlist):
                    GRAD_add_KL[index_save] = np.array(grad_add_KL)
                    GRAD_add_prob[index_save] = np.array(grad_add_prob)
                print('grad_mask_KL', grad_mask_KL)
                print('grad_add_KL', grad_add_KL)
                print('grad_mask_prob', grad_mask_prob)
                print('grad_add_prob', grad_add_prob)


                index_save+=1

        if index % 10 == 0:
            savematrix = dict()
            savematrix['DEEPLIFT_mask_KL'] = DEEPLIFT_mask_KL.tolist()
            savematrix['DEEPLIFT_add_KL'] = DEEPLIFT_add_KL.tolist()
            savematrix['DEEPLIFT_mask_prob]'] = DEEPLIFT_mask_prob.tolist()
            savematrix['DEEPLIFT_add_prob'] = DEEPLIFT_add_prob.tolist()

            savematrix['GRAD_mask_KL'] = GRAD_mask_KL.tolist()
            savematrix['GRAD_add_KL'] = GRAD_add_KL.tolist()
            savematrix['GRAD_mask_prob]'] = GRAD_mask_prob.tolist()
            savematrix['GRAD_add_prob'] = GRAD_add_prob.tolist()

            savematrix['GNNLRP_mask_KL'] = GNNLRP_mask_KL.tolist()
            savematrix['GNNLRP_add_KL'] = GNNLRP_add_KL.tolist()
            savematrix['GNNLRP_mask_prob]'] = GNNLRP_mask_prob.tolist()
            savematrix['GNNLRP_add_prob'] = GNNLRP_add_prob.tolist()

            savematrix['Topk_mask_KL'] = Topk_mask_KL.tolist()
            savematrix['Topk_add_KL'] = Topk_add_KL.tolist()
            savematrix['Topk_mask_prob]'] = Topk_mask_prob.tolist()
            savematrix['Topk_add_prob'] = Topk_add_prob.tolist()

            savematrix['Linear_mask_KL'] = Linear_mask_KL.tolist()
            savematrix['Linear_add_KL'] = Linear_add_KL.tolist()
            savematrix['Linear_mask_prob]'] = Linear_mask_prob.tolist()
            savematrix['Linear_add_prob'] = Linear_add_prob.tolist()

            savematrix['Convex_mask_KL'] = Convex_mask_KL.tolist()
            savematrix['Convex_add_KL'] = Convex_add_KL.tolist()
            savematrix['Convex_mask_prob]'] = Convex_mask_prob.tolist()
            savematrix['Convex_add_prob'] = Convex_add_prob.tolist()

            folder_path = f'result/{args.dataset}'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            json_matrix = json.dumps(savematrix)
            with open(f'result/{args.dataset}/{args.type}_0.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')





    # with open(f'result/{args.dataset}/{args.type}_edgesave.pkl', 'wb') as f:
    #     pickle.dump(save_edge, f)
    # with open(f'result/{args.dataset}/{args.type}_edgesave.pkl', 'rb') as f:
    #     a = pickle.load(f)
    # print(a)
    savematrix = dict()
    savematrix['DEEPLIFT_mask_KL'] = DEEPLIFT_mask_KL.tolist()
    savematrix['DEEPLIFT_add_KL'] = DEEPLIFT_add_KL.tolist()
    savematrix['DEEPLIFT_mask_prob]'] = DEEPLIFT_mask_prob.tolist()
    savematrix['DEEPLIFT_add_prob'] = DEEPLIFT_add_prob.tolist()

    savematrix['GRAD_mask_KL'] = GRAD_mask_KL.tolist()
    savematrix['GRAD_add_KL'] = GRAD_add_KL.tolist()
    savematrix['GRAD_mask_prob]'] = GRAD_mask_prob.tolist()
    savematrix['GRAD_add_prob'] = GRAD_add_prob.tolist()

    savematrix['GNNLRP_mask_KL'] = GNNLRP_mask_KL.tolist()
    savematrix['GNNLRP_add_KL'] = GNNLRP_add_KL.tolist()
    savematrix['GNNLRP_mask_prob]'] = GNNLRP_mask_prob.tolist()
    savematrix['GNNLRP_add_prob'] = GNNLRP_add_prob.tolist()

    savematrix['Topk_mask_KL'] = Topk_mask_KL.tolist()
    savematrix['Topk_add_KL'] = Topk_add_KL.tolist()
    savematrix['Topk_mask_prob]'] = Topk_mask_prob.tolist()
    savematrix['Topk_add_prob'] = Topk_add_prob.tolist()

    savematrix['Linear_mask_KL'] = Linear_mask_KL.tolist()
    savematrix['Linear_add_KL'] = Linear_add_KL.tolist()
    savematrix['Linear_mask_prob]'] = Linear_mask_prob.tolist()
    savematrix['Linear_add_prob'] = Linear_add_prob.tolist()

    savematrix['Convex_mask_KL'] = Convex_mask_KL.tolist()
    savematrix['Convex_add_KL'] = Convex_add_KL.tolist()
    savematrix['Convex_mask_prob]'] = Convex_mask_prob.tolist()
    savematrix['Convex_add_prob'] = Convex_add_prob.tolist()


    savematrix['graphkl']=graph_KL_list
    savematrix['oldprob'] = old_prob.tolist()
    savematrix['newprob'] = new_prob.tolist()
    # print(old_prob_list)
    # print(new_prob_list)


    json_matrix = json.dumps(savematrix)
    with open(f'result/{args.dataset}/{args.type}_0.json',
              'w') as json_file:
        json_file.write(json_matrix)
    print('save success')

    # Gnnexplainer_mask_KL=Gnnexplainer_mask_KL[~(Gnnexplainer_mask_KL == 1).all(1)]
    # Gnnexplainer_add_KL = Gnnexplainer_add_KL[~(Gnnexplainer_add_KL == 1).all(1)]
    # Gnnexplainer_mask_prob = Gnnexplainer_mask_prob[~(Gnnexplainer_mask_prob == 1).all(1)]
    # Gnnexplainer_add_prob = Gnnexplainer_add_prob[~(Gnnexplainer_add_prob == 1).all(1)]
    #
    # Convex_mask_KL = Convex_mask_KL[~(Convex_mask_KL == 1).all(1)]
    # Convex_add_KL = Convex_add_KL[~(Convex_add_KL == 1).all(1)]
    # Convex_mask_prob = Convex_mask_prob[~(Convex_mask_prob == 1).all(1)]
    # Convex_add_prob = Convex_add_prob[~(Convex_add_prob == 1).all(1)]
    #
    # Topk_mask_KL = Topk_mask_KL[~(Topk_mask_KL == 1).all(1)]
    # Topk_add_KL = Topk_add_KL[~(Topk_add_KL == 1).all(1)]
    # Topk_mask_prob = Topk_mask_prob[~(Topk_mask_prob == 1).all(1)]
    # Topk_add_prob = Topk_add_prob[~(Topk_add_prob == 1).all(1)]
    #
    # Linear_mask_KL = Linear_mask_KL[~(Linear_mask_KL == 1).all(1)]
    # Linear_add_KL = Linear_add_KL[~(Linear_add_KL == 1).all(1)]
    # Linear_mask_prob = Linear_mask_prob[~(Linear_mask_prob == 1).all(1)]
    # Linear_add_prob = Linear_add_prob[~(Linear_add_prob == 1).all(1)]
    #
    # GRAD_mask_KL = GRAD_mask_KL[~(GRAD_mask_KL == 1).all(1)]
    # GRAD_add_KL = GRAD_add_KL[~(GRAD_add_KL == 1).all(1)]
    # GRAD_mask_prob = GRAD_mask_prob[~(GRAD_mask_prob == 1).all(1)]
    # GRAD_add_prob = GRAD_add_prob[~(GRAD_add_prob == 1).all(1)]
    #
    # GNNLRP_mask_KL = GNNLRP_mask_KL[~(GNNLRP_mask_KL == 1).all(1)]
    # GNNLRP_add_KL = GNNLRP_add_KL[~(GNNLRP_add_KL == 1).all(1)]
    # GNNLRP_mask_prob = GNNLRP_mask_prob[~(GNNLRP_mask_prob == 1).all(1)]
    # GNNLRP_add_prob = GNNLRP_add_prob[~(GNNLRP_add_prob == 1).all(1)]
    #
    # DEEPLIFT_mask_KL = DEEPLIFT_mask_KL[~(DEEPLIFT_mask_KL == 1).all(1)]
    # DEEPLIFT_add_KL = DEEPLIFT_add_KL[~(DEEPLIFT_add_KL == 1).all(1)]
    # DEEPLIFT_mask_prob = DEEPLIFT_mask_prob[~(DEEPLIFT_mask_prob == 1).all(1)]
    # DEEPLIFT_add_prob = DEEPLIFT_add_prob[~(DEEPLIFT_add_prob == 1).all(1)]
    #
    #
    # Gnnexplainer_mask_KL_mean = np.mean(Gnnexplainer_mask_KL, axis=0)
    # Gnnexplainer_add_KL_mean = np.mean(Gnnexplainer_add_KL, axis=0)
    # Gnnexplainer_mask_prob_mean = np.mean(Gnnexplainer_mask_prob, axis=0)
    # Gnnexplainer_add_prob_mean = np.mean(Gnnexplainer_add_prob, axis=0)
    #
    # Convex_mask_KL_mean = np.mean(Convex_mask_KL, axis=0)
    # Convex_add_KL_mean = np.mean(Convex_add_KL, axis=0)
    # Convex_mask_prob_mean = np.mean(Convex_mask_prob, axis=0)
    # Convex_add_prob_mean = np.mean(Convex_add_prob, axis=0)
    #
    # GNNLRP_mask_KL_mean = np.mean(GNNLRP_mask_KL, axis=0)
    # GNNLRP_add_KL_mean = np.mean(GNNLRP_add_KL, axis=0)
    # GNNLRP_mask_prob_mean = np.mean(GNNLRP_mask_prob, axis=0)
    # GNNLRP_add_prob_mean = np.mean(GNNLRP_add_prob, axis=0)
    #
    # Topk_mask_KL_mean = np.mean(Topk_mask_KL, axis=0)
    # Topk_add_KL_mean = np.mean(Topk_add_KL, axis=0)
    # Topk_mask_prob_mean = np.mean(Topk_mask_prob, axis=0)
    # Topk_add_prob_mean = np.mean(Topk_add_prob, axis=0)
    #
    # Linear_mask_KL_mean = np.mean(Linear_mask_KL, axis=0)
    # Linear_add_KL_mean = np.mean(Linear_add_KL, axis=0)
    # Linear_mask_prob_mean = np.mean(Linear_mask_prob, axis=0)
    # Linear_add_prob_mean = np.mean(Linear_add_prob, axis=0)
    #
    # GRAD_mask_KL_mean = np.mean(GRAD_mask_KL, axis=0)
    # GRAD_add_KL_mean = np.mean(GRAD_add_KL, axis=0)
    # GRAD_mask_prob_mean = np.mean(GRAD_mask_prob, axis=0)
    # GRAD_add_prob_mean = np.mean(GRAD_add_prob, axis=0)
    #
    #
    #
    # DEEPLIFT_mask_KL_mean = np.mean(DEEPLIFT_mask_KL, axis=0)
    # DEEPLIFT_add_KL_mean = np.mean(DEEPLIFT_add_KL, axis=0)
    # DEEPLIFT_mask_prob_mean = np.mean(DEEPLIFT_mask_prob, axis=0)
    # DEEPLIFT_add_prob_mean = np.mean(DEEPLIFT_add_prob, axis=0)
    #
    #
    #
    #
    # x1 = [1, 2, 3, 4, 5]
    #
    # plt.plot(x1, DEEPLIFT_mask_KL_mean, color='black', label='DeepLIFT', ls='--')
    # plt.plot(x1, Convex_mask_KL_mean, color='yellowgreen', label='AxiomPath-Convex', ls='-')
    # plt.plot(x1, Topk_mask_KL_mean, color='blue', label='AxiomPath-Topk', ls='--')
    # plt.plot(x1, Linear_mask_KL_mean, color='orange', label='AxiomPath-Linear', ls='--')
    # plt.plot(x1, GRAD_mask_KL_mean, color='purple', label='Grad', ls='--')
    # plt.plot(x1, GNNLRP_mask_KL_mean, color='green', label='GNN-LRP', ls='--')
    # plt.plot(x1, Gnnexplainer_mask_KL_mean, color='red', label='gnnexplainer', ls='--')
    # plt.legend(loc='best')  # 显示图例
    # plt.title(f'{args.dataset}' + f' {args.type}' + ' mask_kl', fontsize=14)
    # plt.xlabel('Explanation complexity levels', fontsize=14)
    # #
    # plt.ylabel('Fidelity', fontsize=14)
    # plt.show()
    #
    # plt.plot(x1, DEEPLIFT_add_KL_mean, color='black', label='DeepLIFT', ls='--')
    # plt.plot(x1, Convex_add_KL_mean, color='yellowgreen', label='AxiomPath-Convex', ls='-')
    # plt.plot(x1, Topk_add_KL_mean, color='blue', label='AxiomPath-Topk', ls='--')
    # plt.plot(x1, Linear_add_KL_mean, color='orange', label='AxiomPath-Linear', ls='--')
    # plt.plot(x1, GRAD_add_KL_mean, color='purple', label='Grad', ls='--')
    # plt.plot(x1, GNNLRP_add_KL_mean, color='green', label='GNN-LRP', ls='--')
    # plt.plot(x1, Gnnexplainer_add_KL_mean, color='red', label='gnnexplainer', ls='--')
    # plt.legend(loc='best')  # 显示图例
    # plt.title(f'{args.dataset}' + f' {args.type}' + ' add_kl', fontsize=14)
    # plt.xlabel('Explanation complexity levels', fontsize=14)
    # #
    # plt.ylabel('Fidelity', fontsize=14)
    # plt.show()
    #
    # plt.plot(x1, DEEPLIFT_mask_prob_mean, color='black', label='DeepLIFT', ls='--')
    # plt.plot(x1, Convex_mask_prob_mean, color='yellowgreen', label='AxiomPath-Convex', ls='-')
    # plt.plot(x1, Topk_mask_prob_mean, color='blue', label='AxiomPath-Topk', ls='--')
    # plt.plot(x1, Linear_mask_prob_mean, color='orange', label='AxiomPath-Linear', ls='--')
    # plt.plot(x1, GRAD_mask_prob_mean, color='purple', label='Grad', ls='--')
    # plt.plot(x1, GNNLRP_mask_prob_mean, color='green', label='GNN-LRP', ls='--')
    # plt.plot(x1, Gnnexplainer_mask_prob_mean, color='red', label='gnnexplainer', ls='--')
    # plt.legend(loc='best')  # 显示图例
    # plt.title(f'{args.dataset}' + f' {args.type}' + ' mask_prob', fontsize=14)
    # plt.xlabel('Explanation complexity levels', fontsize=14)
    # #
    # plt.ylabel('Fidelity', fontsize=14)
    # plt.show()
    #
    # plt.plot(x1, DEEPLIFT_add_prob_mean, color='black', label='DeepLIFT', ls='--')
    # plt.plot(x1, Convex_add_prob_mean, color='yellowgreen', label='AxiomPath-Convex', ls='-')
    # plt.plot(x1, Topk_add_prob_mean, color='blue', label='AxiomPath-Topk', ls='--')
    # plt.plot(x1, Linear_add_prob_mean, color='orange', label='AxiomPath-Linear', ls='--')
    # plt.plot(x1, GRAD_add_prob_mean, color='purple', label='Grad', ls='--')
    # plt.plot(x1, GNNLRP_add_prob_mean, color='green', label='GNN-LRP', ls='--')
    # plt.plot(x1, Gnnexplainer_add_prob_mean, color='red', label='gnnexplainer', ls='--')
    # plt.legend(loc='best')  # 显示图例
    # plt.title(f'{args.dataset}' + f' {args.type}' + ' add_prob', fontsize=14)
    # plt.xlabel('Explanation complexity levels', fontsize=14)
    # #
    # plt.ylabel('Fidelity', fontsize=14)
    # plt.show()
    #
    #
