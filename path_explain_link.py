from utils.split_data import gen_Yelp_data,gen_link_data
# dynamic_data=gen_dynamic_data('Chi',0,84,90,'month')
# adj_old,adj_new,edges_old,edges_new,graph_old,graph_new,edgelist,_,_=dynamic_data.gen_adj()
# Hold,Hnew,W,model_mask=dynamic_data.gen_parameters()
# print(edgelist)
from  utils.utils_deeplift import findnewpath,dfs2
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
from baselines.gnnlrp_link import gnnlrp_link
from baselines.grad_link import grad_link
import numpy as np
from  utils.utils_deeplift import softmax,KL_divergence
from baselines.convex_link import convex_link
from baselines.deeplift_link import deeplift_link
from baselines.linear_link import linear_link
from baselines.topk_link import topk_link
from utils.evalation import metrics_link
import random
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='UCI') #bitcoinotc bitcoinalpha
    parser.add_argument('--type', type=str, default='both')
    parser.add_argument('--time_index',type=int,default=0)
    parser.add_argument('--start_time1',type=int,default=0)
    parser.add_argument('--end_time1', type=int, default=0)
    parser.add_argument('--start_time2', type=int, default=0)
    parser.add_argument('--end_time2', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--hidden', type=int, default=4)


    args = parser.parse_args()
    time_step=time_step(args.dataset,args.type)
    if args.type=='add':
        args.start_time1=0
        args.start_time2 = 0
        args.end_time1=time_step[args.time_index][0]
        args.end_time2 = time_step[args.time_index][1]
    elif args.type=='remove':
        args.start_time1 = 0
        args.start_time2 = 0
        args.end_time1=time_step[args.time_index][1]
        args.end_time2  = time_step[args.time_index][0]
    else:
        if args.dataset=='bitcoinalpha' or args.dataset=='bitcoinotc':
            args.start_time1=time_step[args.time_index][0]-24
            args.end_time1 = time_step[args.time_index][0]

            args.start_time2 = time_step[args.time_index][1]-24
            args.end_time2 = time_step[args.time_index][1]
        if args.dataset=='UCI':
            args.start_time1 = time_step[args.time_index][0] - 3
            args.end_time1 = time_step[args.time_index][0]

            args.start_time2 = time_step[args.time_index][1] - 3
            args.end_time2 = time_step[args.time_index][1]






    layernumbers=2

    if args.dataset=='UCI':
        flag='week'
    else:
        flag='month'






    dynamic_data=gen_link_data(args.dataset,args.data_path,args.start_time1,args.end_time1,args.start_time2,args.end_time2,flag)
    data,adj_old,adj_new,edges_old,edges_new,edge_index_all,graph_old,graph_new,addedgelist,removeedgelist=dynamic_data.load_data()
    model=dynamic_data.gen_model(data)
    Hold,Hnew,W,W_mlp,Hnew_mlp,Hold_mlp,decode_logits_new,\
    decode_logits_old,decode_label_new=dynamic_data.gen_parameters(model,edge_index_all)
    goal_edge = []
    for i in range(len(decode_label_new)):
        node_KL = KL_divergence(softmax(decode_logits_new.detach().numpy()[i]), softmax(decode_logits_old.detach().numpy()[i]))
        # goal_1=edge_index_all[0][i]
        # goal_2 = edge_index_all[1][i]
        # oldgoalpaths_1 = dfs2(goal_1, goal_1, graph_old, layernumbers + 1, [], [])
        # oldgoalpaths_2 = dfs2(goal_2, goal_2, graph_old, layernumbers + 1, [], [])
        #


        if node_KL > 0.05:
            goal_edge.append((edge_index_all[0][i], edge_index_all[1][i]))
    print('len goal_edge', len(goal_edge))

    if len(goal_edge)>200:
        goal_edge=random.sample(goal_edge,200)
    print('len goal_edge', len(goal_edge))


    delete_list = []
    DEEPLIFT_add_prob = np.ones((len(goal_edge), 5))
    DEEPLIFT_mask_prob = np.ones((len(goal_edge), 5))
    DEEPLIFT_add_KL = np.ones((len(goal_edge), 5))
    DEEPLIFT_mask_KL = np.ones((len(goal_edge), 5))

    Convex_add_prob = np.ones((len(goal_edge), 5))
    Convex_mask_prob = np.ones((len(goal_edge), 5))
    Convex_add_KL = np.ones((len(goal_edge), 5))
    Convex_mask_KL = np.ones((len(goal_edge), 5))

    GNNLRP_add_prob  = np.ones((len(goal_edge), 5))
    GNNLRP_mask_prob = np.ones((len(goal_edge), 5))
    GNNLRP_add_KL = np.ones((len(goal_edge), 5))
    GNNLRP_mask_KL = np.ones((len(goal_edge), 5))

    GRAD_add_prob = np.ones((len(goal_edge), 5))
    GRAD_mask_prob = np.ones((len(goal_edge), 5))
    GRAD_add_KL = np.ones((len(goal_edge), 5))
    GRAD_mask_KL = np.ones((len(goal_edge), 5))

    Topk_add_prob= np.ones((len(goal_edge), 5))
    Topk_mask_prob = np.ones((len(goal_edge), 5))
    Topk_add_KL= np.ones((len(goal_edge), 5))
    Topk_mask_KL = np.ones((len(goal_edge), 5))

    Linear_add_prob = np.ones((len(goal_edge), 5))
    Linear_mask_prob = np.ones((len(goal_edge), 5))
    Linear_add_KL = np.ones((len(goal_edge), 5))
    Linear_mask_KL = np.ones((len(goal_edge), 5))

    Gnnexplainer_add_prob = np.ones((len(goal_edge), 5))
    Gnnexplainer_mask_prob = np.ones((len(goal_edge), 5))
    Gnnexplainer_add_KL= np.ones((len(goal_edge), 5))
    Gnnexplainer_mask_KL=np.ones((len(goal_edge), 5))

    edges_dict_all = dict()
    for idx, node in enumerate(edge_index_all[0]):
        edges_dict_all[(node, edge_index_all[1][idx])] = idx
    #
    for goal_edge_index in range(len(goal_edge)):

        index=goal_edge_index
        print('index',index)
        index_new = edges_dict_all[goal_edge[goal_edge_index]]
        index_old = index_new
        goal_1 = goal_edge[goal_edge_index][0]
        goal_2 = goal_edge[goal_edge_index][1]
        addgoalpath_1 = findnewpath(addedgelist, graph_new, layernumbers, goal_1)

        addgoalpath_2 = findnewpath(addedgelist, graph_new, layernumbers, goal_2)
        removegoalpath_1 =findnewpath(removeedgelist, graph_old, layernumbers, goal_1)
        removegoalpath_2 = findnewpath(removeedgelist, graph_old, layernumbers, goal_2)

        newgoalpaths_1 = dfs2(goal_1, goal_1,graph_new, layernumbers + 1, [], [])
        oldgoalpaths_1 = dfs2(goal_1, goal_1, graph_old, layernumbers + 1, [], [])

        newgoalpaths_2 = dfs2(goal_2, goal_2, graph_new, layernumbers + 1, [], [])
        oldgoalpaths_2 = dfs2(goal_2, goal_2, graph_old, layernumbers + 1, [], [])
        print('oldgoalpaths_1',len(oldgoalpaths_1))
        print('oldgoalpaths_2',len(oldgoalpaths_2))
        print('newgoalpaths_1',len(newgoalpaths_1))
        print('newgoalpaths_2', len(newgoalpaths_2))

        print('addgoalpath_1',len(addgoalpath_1))
        print('addgoalpath_2', len(addgoalpath_2))

        print('removegoalpath_1',len(removegoalpath_1))
        print('removegoalpath_2', len(removegoalpath_2))
        # if goal_1!=goal_2:
        #     for path in removegoalpath_1:
        #         if path in removegoalpath_2:
        #             print(path, goal_1, goal_2)
        #             print('remove yes')
        #     for path in addgoalpath_1:
        #         if path in addgoalpath_2:
        #             print(path, goal_1, goal_2)
        #             print('add yes')
        #
        #     print(Hold[layernumbers * 2 - 1][goal_1])
        #     print(Hold[layernumbers * 2 - 1][goal_2])
        #     select_pathlist = path_number(args.dataset, args.type,
        #                                   len(addgoalpath_1) + len(addgoalpath_2) + len(removegoalpath_2) + len(
        #                                       removegoalpath_1))
        #
        #     pred_old = metrics_link(model, data[0].x, goal_1, goal_2,
        #                             addgoalpath_1 + addgoalpath_2, removegoalpath_1 + removegoalpath_2, edges_new,
        #                             args.hidden, Hnew,'mask',removeedgelist,addedgelist).cal()
        #     print('pred_old', pred_old)
        #     print('true hold', Hold_mlp[1][index_old])
        #     pred_new = metrics_link(model, data[0].x, goal_1, goal_2,
        #                             addgoalpath_1 + addgoalpath_2, removegoalpath_1 + removegoalpath_2,edges_old, \
        #                             args.hidden, Hold,'add',removeedgelist,addedgelist).cal()
        #     print('pred_new', pred_new)
        #     print('true hnew', Hnew_mlp[1][index_old])




        select_pathlist =path_number(args.dataset,args.type,len(addgoalpath_1) + len(addgoalpath_2) + len(removegoalpath_2) + len(
                removegoalpath_1))
        # KL_eval = metrics_KL(Hold, Hnew, goal, layernumbers)
        # prob_eval = metrics_prob(Hold, Hnew, goal, layernumbers)
        edge_kl= KL_divergence(softmax((Hnew_mlp[1][index_new])),
                                 softmax((Hold_mlp[1][index_old])))

        if 10 < (len(addgoalpath_1) + len(addgoalpath_2) + len(removegoalpath_2) + len(
                removegoalpath_1)) and goal_1 != goal_2 and len(oldgoalpaths_1)+len(oldgoalpaths_2)>1:
            print('index',index)

            print('len path', len(addgoalpath_1) + len(addgoalpath_2) + len(removegoalpath_2) + len(removegoalpath_1))
            print('KL', edge_kl)
            KL_eval = metrics_KL(Hold_mlp, Hnew_mlp, index_old, 1)
            prob_eval = metrics_prob(Hold_mlp, Hnew_mlp, index_new, 1)



            # grad_method = grad_link(Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, \
            #                             index_new, index_old, layernumbers, args.hidden, data, graph_new, graph_old, \
            #                             addgoalpath_1, addgoalpath_2, removegoalpath_1, removegoalpath_2,
            #                             oldgoalpaths_1, \
            #                             oldgoalpaths_2, newgoalpaths_1, newgoalpaths_2, select_pathlist, edges_new,
            #                             edges_old, model,adj_old,adj_new)
            # print(adj_old)
            # grad_logists_mask, grad_logists_add =grad_method.select_importantpath()
            #
            # print(grad_logists_mask)

            convex_method=convex_link(Hold,Hnew,W,Hnew_mlp,Hold_mlp,W_mlp,goal_1,goal_2,\
                                      index_new,index_old,layernumbers,args.hidden,data,graph_new,graph_old,\
                                      addgoalpath_1,addgoalpath_2,removegoalpath_1,removegoalpath_2,\
                                      select_pathlist,edges_new,edges_old,model,addedgelist,removeedgelist)
            contriution_value = convex_method.contribution_value()

            print('p',sum(contriution_value))
            print('true', Hnew_mlp[1][index_new] - Hold_mlp[1][index_old])
            print('Hold', Hold_mlp[1][index_old])
            print('Hnew', Hnew_mlp[1][index_new])


            convex_logists_mask, convex_logists_add =convex_method.select_importantpath(Hold,Hnew,removeedgelist,addedgelist)
            # print(convex_logists_mask)
            convex_mask_KL, convex_add_KL = KL_eval.KL(convex_logists_add, convex_logists_mask)
            convex_mask_prob, convex_add_prob = prob_eval.prob(convex_logists_add, convex_logists_mask)
            Convex_mask_KL[index] = np.array(convex_mask_KL)
            Convex_add_KL[index] = np.array(convex_add_KL)
            Convex_mask_prob[index] = np.array(convex_mask_prob)
            Convex_add_prob[index] = np.array(convex_add_prob)
            print('convex_mask_KL', convex_mask_KL)
            print('convex_add_KL', convex_add_KL)
            print('convex_mask_prob', convex_mask_prob)
            print('convex_add_prob', convex_add_prob)

            deeplift_method = deeplift_link(Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, \
                                        index_new, index_old, layernumbers, args.hidden, data, graph_new, graph_old, \
                                        addgoalpath_1, addgoalpath_2, removegoalpath_1, removegoalpath_2, \
                                        select_pathlist, edges_new, edges_old, model, addedgelist, removeedgelist)
            deeplift_logists_mask, deeplift_logists_add = deeplift_method.select_importantpath(contriution_value,Hold,Hnew,removeedgelist,addedgelist)
            deeplift_mask_KL, deeplift_add_KL = KL_eval.KL(deeplift_logists_add, deeplift_logists_mask)
            deeplift_mask_prob, deeplift_add_prob = prob_eval.prob(deeplift_logists_add, deeplift_logists_mask)

            DEEPLIFT_mask_KL[index] = np.array(deeplift_mask_KL)
            DEEPLIFT_add_KL[index] = np.array(deeplift_add_KL)
            DEEPLIFT_mask_prob[index] = np.array(deeplift_mask_prob)
            DEEPLIFT_add_prob[index] = np.array(deeplift_add_prob)
            print('deeplift_mask_KL', deeplift_mask_KL)
            print('deeplift_add_KL', deeplift_add_KL)
            print('deeplift_mask_prob', deeplift_mask_prob)
            print('deeplift_add_prob', deeplift_add_prob)



            linear_method = linear_link(Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, \
                                        index_new, index_old, layernumbers, args.hidden, data, graph_new, graph_old, \
                                        addgoalpath_1, addgoalpath_2, removegoalpath_1, removegoalpath_2, \
                                        select_pathlist, edges_new, edges_old, model, addedgelist, removeedgelist)
            linear_logists_mask, linear_logists_add = linear_method.select_importantpath(contriution_value,Hold,Hnew, removeedgelist, addedgelist)
            linear_mask_KL, linear_add_KL = KL_eval.KL(linear_logists_add, linear_logists_mask)
            linear_mask_prob, linear_add_prob = prob_eval.prob(linear_logists_add, linear_logists_mask)

            Linear_mask_KL[index] = np.array(linear_mask_KL)
            Linear_add_KL[index] = np.array(linear_add_KL)
            Linear_mask_prob[index] = np.array(linear_mask_prob)
            Linear_add_prob[index] = np.array(linear_add_prob)

            print('linear_mask_KL', linear_mask_KL)
            print('linear_add_KL', linear_add_KL)
            print('linear_mask_prob', linear_mask_prob)
            print('linear_add_prob', linear_add_prob)

            topk_method = topk_link(Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, \
                                        index_new, index_old, layernumbers, args.hidden, data, graph_new, graph_old, \
                                        addgoalpath_1, addgoalpath_2, removegoalpath_1, removegoalpath_2, \
                                        select_pathlist, edges_new, edges_old, model, addedgelist, removeedgelist)
            topk_logists_mask, topk_logists_add = topk_method.select_importantpath(contriution_value,Hold,Hnew, removeedgelist, addedgelist)
            topk_mask_KL, topk_add_KL = KL_eval.KL(topk_logists_add, topk_logists_mask)
            topk_mask_prob, topk_add_prob = prob_eval.prob(topk_logists_add, topk_logists_mask)

            Topk_mask_KL[index] = np.array(topk_mask_KL)
            Topk_add_KL[index] = np.array(topk_add_KL)
            Topk_mask_prob[index] = np.array(topk_mask_prob)
            Topk_add_prob[index] = np.array(topk_add_prob)

            print('topk_mask_KL', topk_mask_KL)
            print('topk_add_KL', topk_add_KL)
            print('topk_mask_prob', topk_mask_prob)
            print('topk_add_prob', topk_add_prob)

            gnnlrp_method=gnnlrp_link(Hold,Hnew,W,Hnew_mlp,Hold_mlp,W_mlp,goal_1,goal_2,\
                                      index_new,index_old,layernumbers,args.hidden,data,graph_new,graph_old,\
                                      addgoalpath_1,addgoalpath_2,removegoalpath_1,removegoalpath_2,oldgoalpaths_1,\
                                      oldgoalpaths_2,newgoalpaths_1,newgoalpaths_2,select_pathlist,edges_new,edges_old,model)
            gnnlrp_logists_mask, gnnlrp_logists_add = gnnlrp_method.select_importantpath(removeedgelist, addedgelist)
            print(gnnlrp_logists_mask, gnnlrp_logists_add)

            gnnlrp_mask_KL, gnnlrp_add_KL = KL_eval.KL(gnnlrp_logists_add, gnnlrp_logists_mask)

            gnnlrp_mask_prob, gnnlrp_add_prob = prob_eval.prob(gnnlrp_logists_add, gnnlrp_logists_mask)
            if len(gnnlrp_mask_KL) == 5:
                GNNLRP_mask_KL[index] = np.array(gnnlrp_mask_KL)
                GNNLRP_mask_prob[index] = np.array(gnnlrp_mask_prob)
            if len(gnnlrp_add_prob)==5:
                GNNLRP_add_KL[index] = np.array(gnnlrp_add_KL)
                GNNLRP_add_prob[index] = np.array(gnnlrp_add_prob)




            print('gnnlrp_mask_KL', gnnlrp_mask_KL)
            print('gnnlrp_add_KL', gnnlrp_add_KL)
            print('gnnlrp_mask_prob', gnnlrp_mask_prob)
            print('gnnlrp_add_prob', gnnlrp_add_prob)

            grad_method = grad_link(Hold, Hnew, W, Hnew_mlp, Hold_mlp, W_mlp, goal_1, goal_2, \
                                        index_new, index_old, layernumbers, args.hidden, data, graph_new, graph_old, \
                                        addgoalpath_1, addgoalpath_2, removegoalpath_1, removegoalpath_2,
                                        oldgoalpaths_1, \
                                        oldgoalpaths_2, newgoalpaths_1, newgoalpaths_2, select_pathlist, edges_new,
                                        edges_old, model,adj_old,adj_new)

            grad_logits_mask, grad_logits_add =grad_method.select_importantpath(removeedgelist, addedgelist)
            grad_mask_KL, grad_add_KL = KL_eval.KL(grad_logits_add, grad_logits_mask)
            grad_mask_prob, grad_add_prob = prob_eval.prob(grad_logits_add, grad_logits_mask)
            if len(grad_mask_KL) == 5:
                GRAD_mask_KL[index] = np.array(grad_mask_KL)
                GRAD_mask_prob[index] = np.array(grad_mask_prob)
            if len(grad_add_prob) == 5:
                GRAD_add_KL[index] = np.array(grad_add_KL)
                GRAD_add_prob[index] = np.array(grad_add_prob)

            print('grad_mask_KL', grad_mask_KL)
            print('grad_add_KL', grad_add_KL)
            print('grad_mask_prob', grad_mask_prob)
            print('grad_add_prob', grad_add_prob)


        if index%10==0:
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

            savematrix['Gnnexplainer_mask_KL'] = Gnnexplainer_mask_KL.tolist()
            savematrix['Gnnexplainer_add_KL'] = Gnnexplainer_add_KL.tolist()
            savematrix['Gnnexplainer_mask_prob]'] = Gnnexplainer_mask_prob.tolist()
            savematrix['Gnnexplainer_add_prob'] = Gnnexplainer_add_prob.tolist()
            savematrix['delete_list'] = delete_list

            folder_path = f'result/{args.dataset}'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            json_matrix = json.dumps(savematrix)
            with open(f'result/{args.dataset}/{args.type}_{args.time_index}.json',
                      'w') as json_file:
                json_file.write(json_matrix)
            print('save success')
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

    savematrix['Gnnexplainer_mask_KL'] = Gnnexplainer_mask_KL.tolist()
    savematrix['Gnnexplainer_add_KL'] = Gnnexplainer_add_KL.tolist()
    savematrix['Gnnexplainer_mask_prob]'] = Gnnexplainer_mask_prob.tolist()
    savematrix['Gnnexplainer_add_prob'] = Gnnexplainer_add_prob.tolist()
    savematrix['delete_list'] = delete_list
    # savematrix['goallist'] = clear_goallist

    json_matrix = json.dumps(savematrix)
    with open(f'result/{args.dataset}/{args.type}_{args.time_index}.json',
              'w') as json_file:
        json_file.write(json_matrix)
    print('save success')













