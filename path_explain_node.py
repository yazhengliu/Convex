from utils.split_data import gen_Yelp_data
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
from baselines.deeplift import deeplift,softmax,KL_divergence
from utils.evalation import metrics_KL,metrics_prob
import random
import numpy as np
from utils.evalation import metrics
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Chi')
    parser.add_argument('--type', type=str, default='both')
    parser.add_argument('--time_index',type=int,default=3)
    parser.add_argument('--start_time1',type=int,default=0)
    parser.add_argument('--end_time1', type=int, default=0)
    parser.add_argument('--start_time2', type=int, default=0)
    parser.add_argument('--end_time2', type=int, default=0)



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
        if args.dataset=='Chi':
            args.start_time1=time_step[args.time_index][0]-6
            args.end_time1 = time_step[args.time_index][0]

            args.start_time2 = time_step[args.time_index][1]
            args.end_time2 = time_step[args.time_index][1]+6
        elif args.dataset=='NYC':
            args.start_time1 = time_step[args.time_index][0] - 6
            args.end_time1 = time_step[args.time_index][0]

            args.start_time2 = time_step[args.time_index][1]
            args.end_time2 = time_step[args.time_index][1] + 6
        else:
            args.start_time1 = time_step[args.time_index][0] - 16
            args.end_time1 = time_step[args.time_index][0]

            args.start_time2 = time_step[args.time_index][1]
            args.end_time2 = time_step[args.time_index][1] + 16




    layernumbers=2

    if args.dataset=='Zip' and args.type=='both':
        flag='week'
    else:
        flag='month'



    if args.type=='add' or args.type=='remove':
        jsonPath = f'data/goal/{args.dataset}/goal_list_'+str(time_step[args.time_index][0])+'_'+str(time_step[args.time_index][1])+ '.json'
        with open(jsonPath, 'r') as f:
            goal_list = json.load(f)
        #print(len(goal_list))
    else:
        jsonPath = f'data/goal/{args.dataset}/both_goal_list_' + str(time_step[args.time_index][0]) + '_' + str(
            time_step[args.time_index][1]) + '.json'
        with open(jsonPath, 'r') as f:
            goal_list = json.load(f)


    # if len(clear_goallist)>500:
    #     clear_goallist=random.sample(clear_goallist,500)
    # print('len clear_goallist', len(clear_goallist))
    dynamic_data=gen_Yelp_data(args.dataset,args.start_time1,args.end_time1,args.start_time2,args.end_time2,flag)
    adj_old,adj_new,edges_old,edges_new,graph_old,graph_new,addedgelist,removeedgelist,_,_=dynamic_data.gen_adj()
    Hold,Hnew,W,features_clear,model_mask,model_gnn=dynamic_data.gen_parameters()

    clear_goallist=[]
    for key, value in goal_list.items():
        goal=int(key)
        oldgoalpaths = dfs2(goal, goal, graph_old, layernumbers + 1, [], [])
        newgoalpaths = dfs2(goal, goal, graph_new, layernumbers + 1, [], [])

        if float(
                value) > 0.05 and (len(oldgoalpaths)>1 and len(newgoalpaths)>1):
            # clear_goallist.append(int(key))# and (Hold[layernumbers*2-1][int(key)][0]!=0 or Hold[layernumbers*2-1][int(key)][1]!=0):
            addgoalpath = findnewpath(addedgelist, graph_new, layernumbers, goal)

            removegoalpath = findnewpath(removeedgelist, graph_old, layernumbers, goal)
            if len(addgoalpath)+len(removegoalpath)>10:
                clear_goallist.append(int(key))
    print('len clear_goallist', len(clear_goallist))

    if len(clear_goallist)>4000:
        clear_goallist=random.sample(clear_goallist,4000)
    print('len clear_goallist', len(clear_goallist))


    delete_list = []
    DEEPLIFT_add_prob = np.ones((len(clear_goallist), 5))
    DEEPLIFT_mask_prob = np.ones((len(clear_goallist), 5))
    DEEPLIFT_add_KL = np.ones((len(clear_goallist), 5))
    DEEPLIFT_mask_KL = np.ones((len(clear_goallist), 5))

    Convex_add_prob = np.ones((len(clear_goallist), 5))
    Convex_mask_prob = np.ones((len(clear_goallist), 5))
    Convex_add_KL = np.ones((len(clear_goallist), 5))
    Convex_mask_KL = np.ones((len(clear_goallist), 5))

    GNNLRP_add_prob  = np.ones((len(clear_goallist), 5))
    GNNLRP_mask_prob = np.ones((len(clear_goallist), 5))
    GNNLRP_add_KL = np.ones((len(clear_goallist), 5))
    GNNLRP_mask_KL = np.ones((len(clear_goallist), 5))

    GRAD_add_prob = np.ones((len(clear_goallist), 5))
    GRAD_mask_prob = np.ones((len(clear_goallist), 5))
    GRAD_add_KL = np.ones((len(clear_goallist), 5))
    GRAD_mask_KL = np.ones((len(clear_goallist), 5))

    Topk_add_prob= np.ones((len(clear_goallist), 5))
    Topk_mask_prob = np.ones((len(clear_goallist), 5))
    Topk_add_KL= np.ones((len(clear_goallist), 5))
    Topk_mask_KL = np.ones((len(clear_goallist), 5))

    Linear_add_prob = np.ones((len(clear_goallist), 5))
    Linear_mask_prob = np.ones((len(clear_goallist), 5))
    Linear_add_KL = np.ones((len(clear_goallist), 5))
    Linear_mask_KL = np.ones((len(clear_goallist), 5))

    Gnnexplainer_add_prob = np.ones((len(clear_goallist), 5))
    Gnnexplainer_mask_prob = np.ones((len(clear_goallist), 5))
    Gnnexplainer_add_KL= np.ones((len(clear_goallist), 5))
    Gnnexplainer_mask_KL=np.ones((len(clear_goallist), 5))
    # clear_goallist=[clear_goallist[147]]
    goal_kl_list=[]

    for goal in clear_goallist:
        index = clear_goallist.index(goal)

        addgoalpath =findnewpath(addedgelist, graph_new, layernumbers, goal)
        removegoalpath = findnewpath(removeedgelist, graph_old, layernumbers, goal)
        select_pathlist =path_number(args.dataset,args.type,len(addgoalpath)+len(removegoalpath))
        select_pathlist=[select_pathlist[0]]

        KL_eval = metrics_KL(Hold, Hnew, goal, layernumbers)
        prob_eval = metrics_prob(Hold, Hnew, goal, layernumbers)
        goal_kl = KL_divergence(softmax((Hnew[2*layernumbers-1][goal])),
                                softmax((Hold[2*layernumbers-1][goal])))
        print('goal_kl',goal_kl)
        goal_kl_list.append(goal_kl)

        if len(addgoalpath)+len(removegoalpath)<=10:
            delete_list.append(clear_goallist.index(goal))

        print('len paths', len(addgoalpath) + len(removegoalpath))
        if addgoalpath == []:
            print('remove paths need to be added')
        if removegoalpath == []:
            print('add paths need to be  mask')

        if addgoalpath != [] and removegoalpath != []:
            print('both')
        if len(addgoalpath) + len(removegoalpath) > 10:
            newgoalpaths = dfs2(goal, goal, graph_new, layernumbers + 1, [], [])
            oldgoalpaths = dfs2(goal, goal, graph_old, layernumbers + 1, [], [])

            index = clear_goallist.index(goal)
            print('index', index)

            # pre_old=metrics(model_mask, features_clear, goal, addgoalpath, removegoalpath, edges_new,
            #         args.dataset).cal()
            # pre_new=metrics(model_mask, features_clear, goal,removegoalpath, addgoalpath, edges_old,
            #         args.dataset).cal()
            # print('Hold', Hold[layernumbers * 2 - 1][goal])
            # print('pre_old',pre_old)
            # print('Hnew', Hnew[layernumbers * 2 - 1][goal])
            # print('pre_new',pre_new)



            convex_method=convex(Hold,Hnew,W,goal,addedgelist,removeedgelist,addgoalpath,removegoalpath,\
                                 features_clear,layernumbers,2,select_pathlist,\
                                 model_mask,edges_new,edges_old,args.dataset,args.type)

            contriution_value=convex_method.contribution_value()

            print('p',sum(contriution_value))
            print('true',Hnew[layernumbers*2-1][goal]-Hold[layernumbers*2-1][goal])
            print('Hold',Hold[layernumbers*2-1][goal])
            print('Hnew',Hnew[layernumbers * 2 - 1][goal])
            convex_logists_mask,convex_logists_add=convex_method.select_importantpath(contriution_value,removeedgelist,addedgelist)
            print(convex_logists_mask[0])
            print(convex_logists_add[0])
            convex_mask_KL, convex_add_KL = KL_eval.KL(convex_logists_add, convex_logists_mask)
            convex_mask_prob, convex_add_prob = prob_eval.prob(convex_logists_add, convex_logists_mask)

            Convex_mask_KL[index]=np.array(convex_mask_KL)
            Convex_add_KL[index] = np.array(convex_add_KL)
            Convex_mask_prob[index] = np.array(convex_mask_prob)
            Convex_add_prob[index] = np.array(convex_add_prob)


            print('convex_mask_KL',convex_mask_KL)
            print('convex_add_KL',convex_add_KL)
            print('convex_mask_prob',convex_mask_prob)
            print('convex_add_prob',convex_add_prob)


            deeplift_method=deeplift(Hold,Hnew,W,goal,addgoalpath,removegoalpath,features_clear,\
                                     layernumbers,select_pathlist,model_mask,edges_new,edges_old,args.dataset)
            deeplift_logists_mask,deeplift_logists_add=deeplift_method.select_importantpath(contriution_value,removeedgelist,addedgelist)
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

            topk_method=topk(goal,addgoalpath,removegoalpath,features_clear,layernumbers,select_pathlist,model_mask,edges_new,edges_old,args.dataset)
            topk_logists_mask,topk_logists_add=topk_method.select_importantpath(contriution_value,removeedgelist,addedgelist)
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

            #
            linear_method=linear(Hold,Hnew,W,goal,addgoalpath,removegoalpath,features_clear,layernumbers,select_pathlist,\
                                 model_mask,edges_new,2,edges_old,args.dataset)
            linear_logists_mask,linear_logists_add=linear_method.select_importantpath(contriution_value,removeedgelist, addedgelist)

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




            gnnlrp_method=gnnlrp(Hold,Hnew,W,goal,addgoalpath,removegoalpath,features_clear,layernumbers\
                                 ,select_pathlist,model_mask,edges_new,graph_new,graph_old,edges_old,args.dataset,newgoalpaths,oldgoalpaths)
            gnnlrp_logists_mask,gnnlrp_logists_add=gnnlrp_method.select_importantpath(removeedgelist, addedgelist)
            gnnlrp_mask_KL, gnnlrp_add_KL = KL_eval.KL(gnnlrp_logists_add, gnnlrp_logists_mask)
            gnnlrp_mask_prob, gnnlrp_add_prob = prob_eval.prob(gnnlrp_logists_add, gnnlrp_logists_mask)
            old_index = np.argmax(Hold[2 * layernumbers - 1][goal])
            new_index = np.argmax(Hnew[2 * layernumbers - 1][goal])
            if len(gnnlrp_mask_KL) == 5:
                GNNLRP_mask_KL[index] = np.array(gnnlrp_mask_KL)
                GNNLRP_mask_prob[index] = np.array(gnnlrp_mask_prob)

            if len(gnnlrp_add_KL)==5:
                GNNLRP_add_KL[index] = np.array(gnnlrp_add_KL)
                GNNLRP_add_prob[index] = np.array(gnnlrp_add_prob)
            print('gnnlrp_mask_KL', gnnlrp_mask_KL)
            print('gnnlrp_add_KL', gnnlrp_add_KL)
            print('gnnlrp_mask_prob', gnnlrp_mask_prob)
            print('gnnlrp_add_prob', gnnlrp_add_prob)


            grad_method=grad(Hold,Hnew,W,goal,addgoalpath,removegoalpath,features_clear,layernumbers,\
                             select_pathlist,model_mask,edges_new,edges_old,graph_new,graph_old,args.dataset,\
                             newgoalpaths,oldgoalpaths)
            grad_logits_mask,grad_logits_add=grad_method.logits()
            print(grad_logits_add)
            grad_mask_KL,grad_add_KL=KL_eval.KL(grad_logits_add,grad_logits_mask)
            grad_mask_prob, grad_add_prob = prob_eval.prob(grad_logits_add, grad_logits_mask)
            if len(grad_mask_KL)==5:
                GRAD_mask_KL[index] = np.array(grad_mask_KL)
                GRAD_mask_prob[index] = np.array(grad_mask_prob)
            if len(grad_add_prob)==5:
                GRAD_add_KL[index] = np.array(grad_add_KL)
                GRAD_add_prob[index] = np.array(grad_add_prob)



            print('grad_mask_KL', grad_mask_KL)
            print('grad_add_KL', grad_add_KL)
            print('grad_mask_prob', grad_mask_prob)
            print('grad_add_prob', grad_add_prob)



            gnnexplainer_method=gnnexplainer(model_gnn,goal,features_clear,edges_new,edges_old,graph_new,graph_old,layernumbers,\
                                             model_mask,select_pathlist,addgoalpath,removegoalpath,args.dataset)
            gnnexplainer_logists_mask,gnnexplainer_logists_add=gnnexplainer_method.select_importantpath(removeedgelist,addedgelist)

            gnnexplainer_mask_KL, gnnexplainer_add_KL = KL_eval.KL(gnnexplainer_logists_add, gnnexplainer_logists_mask)
            gnnexplainer_mask_prob, gnnexplainer_add_prob = prob_eval.prob(gnnexplainer_logists_add, gnnexplainer_logists_mask)

            if len(gnnexplainer_mask_KL)==5:
                print('yes')
                Gnnexplainer_mask_KL[index] = np.array(gnnexplainer_mask_KL)
                Gnnexplainer_mask_prob[index] = np.array(gnnexplainer_mask_prob)
            if len(gnnexplainer_add_KL)==5:
                print('yes')
                Gnnexplainer_add_KL[index] = np.array(gnnexplainer_add_KL)
                Gnnexplainer_add_prob[index] = np.array(gnnexplainer_add_prob)
            print('gnnexplainer_mask_KL', gnnexplainer_mask_KL)
            print('gnnexplainer_add_KL', gnnexplainer_add_KL)
            print('gnnexplainer_mask_prob', gnnexplainer_mask_prob)
            print('gnnexplainer_add_prob', gnnexplainer_add_prob)

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
            savematrix['goallist'] = clear_goallist
            savematrix['goalkl'] = goal_kl_list

            folder_path=f'result/{args.dataset}'

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
    savematrix['goallist'] = clear_goallist
    savematrix['goalkl'] = goal_kl_list

    json_matrix = json.dumps(savematrix)
    with open(f'result/{args.dataset}/{args.type}_{args.time_index}.json',
              'w') as json_file:
        json_file.write(json_matrix)
    print('save success')













