import numpy as np
import cvxpy as cvx
import torch
import copy
from torch import Tensor
import scipy.sparse as sp
def construct_edge(ground_truth, idx_map, labels, rev_time, time1, time2, flag):
    edges = [[], []]

    # print(ground_truth.keys())
    keys_list = list(ground_truth.keys())
    if flag == 'month':
        for it, r_id in enumerate(ground_truth.keys()):
            if rev_time[r_id][1] >= time1 and rev_time[r_id][1] < time2:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
                # edges.append((idx_map[r_id], idx_map[r_id[0]]))
                # edges.append((idx_map[r_id], idx_map[r_id[1]]))
                #
                # edges.append((idx_map[r_id[0]], idx_map[r_id]))
                # edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'week':
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][2] < time2 and rev_time[r_id][2] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if  rev_time[r_id][2] >= time1 and rev_time[r_id][2]<time2 :
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    if flag == 'year':
        # for i in range(0, len(keys_list)):
        #     r_id = keys_list[i]
        #     if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
        #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
        #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
        #
        #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
        #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
        for it, r_id in enumerate(ground_truth.keys()):
            # print('r_id',r_id)
            if rev_time[r_id][0] < time2 and rev_time[r_id][0] >= time1:
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[0]])
                edges[0].append(idx_map[r_id])
                edges[1].append(idx_map[r_id[1]])
                edges[0].append(idx_map[r_id[1]])
                edges[1].append(idx_map[r_id])
                edges[0].append(idx_map[r_id[0]])
                edges[1].append(idx_map[r_id])
    # for i in range(0,len(keys_list)):
    #     r_id=keys_list[i]
    #     if rev_time[r_id]< year2 and rev_time[r_id]>=year1:
    #         edges.append((idx_map[r_id], idx_map[r_id[0]]))
    #         edges.append((idx_map[r_id], idx_map[r_id[1]]))
    #
    #         edges.append((idx_map[r_id[0]], idx_map[r_id]))
    #         edges.append((idx_map[r_id[1]], idx_map[r_id]))
    edgeitself = list(range(labels.shape[0]))
    for it, edge in enumerate(edgeitself):
        edges[0].append(edge)
        edges[1].append(edge)

    return edges
def matrixtodict(nonzero): # 将邻接矩阵变为字典形式存储
    a = []
    graph = dict()
    for i in range(0, len(nonzero[1])):
        if i != len(nonzero[1]) - 1:
            if nonzero[0][i] == nonzero[0][i + 1]:
                a.append(nonzero[1][i])
            if nonzero[0][i] != nonzero[0][i + 1]:
                a.append(nonzero[1][i])
                graph[nonzero[0][i]] = a
                a = []
        if i == len(nonzero[1]) - 1:
            a.append(nonzero[1][i])
        graph[nonzero[0][len(nonzero[1]) - 1]] = a
    return graph
def difference(edgeindex1,edgeindex2): #tensor
    edgedict1=dict()
    edgedict2 = dict()
    in1_not2=[]
    in2_not1=[]
    for idx,node in enumerate(edgeindex1[0]):
        # print(node)
        # print(edgeindex1[1][idx])
        edgedict1[node,edgeindex1[1][idx]]=idx
    for idx,node in enumerate(edgeindex2[0]):
        edgedict2[node,edgeindex2[1][idx]]=idx
    for key in edgedict1.keys():
        if key not in edgedict2.keys():
            in1_not2.append(key)
    for key in edgedict2.keys():
        if key not in edgedict1.keys():
            in2_not1.append(key)
    # print('in1_not2',in1_not2)
    # print('in2_not1', in2_not1)
    print(len(in1_not2))
    print(len(in2_not1))
    return in1_not2,in2_not1
def clear(edges):
    edge_clear=[]
    for idx,edge in enumerate(edges):
        if idx%1000==0:
            print('idx',idx)
        if [edge[0],edge[1]] not in edge_clear and [edge[1],edge[0]] not in edge_clear:
            edge_clear.append([edge[0],edge[1]])
    return edge_clear
def dfs(start,index,end,graph,length,path=[],paths=[]):#找到起点到终点的全部路径
    path.append(index)
    if len(path)==length:
        if path[-1]==end:
            paths.append(path.copy())
            path.pop()
        else:
            path.pop()

    else:
        for item in graph[index]:
            # if item not in path:
                dfs(start,item,end,graph,length,path,paths)
        path.pop()
    return paths
def dfs2(start,index,graph,length,path=[],paths=[]):#找到起点长度定值的全部路径
    path.append(index)
    # print('index',index)
    # if length==0:
    #     return paths
    if len(path)==length:
        paths.append(path.copy())
        path.pop()
    else:
        for item in graph[index]:
            # if item not in path:
                dfs2(start,item,graph,length,path,paths)
        path.pop()

    return paths
def findnewpath(addedgelist,graph,layernumbers,goal):
    resultpath=[]
    for edge in addedgelist:
        # print(edge)
        if edge[0] == goal:
            pt5 = dfs2(edge[1], edge[1], graph, layernumbers, [], [])
            # print(pt5)
            for i1 in pt5:
                # print(i1)
                i1.pop(0)
                if [goal, edge[1]] + i1 not in resultpath:
                    resultpath.append([goal, edge[1]] + i1)

        if edge[1] == goal:
            pt6 = dfs2(edge[0], edge[0], graph, layernumbers, [], [])
            for i1 in pt6:
                i1.pop(0)
                if [goal, edge[0]] + i1 not in resultpath:
                    resultpath.append([goal, edge[0]] + i1)

        for i in range(0, layernumbers - 1):
            # print('i', i)
            pt1 = dfs(goal, goal, edge[0], graph, i + 2, [], [])
            pt2 = dfs2(edge[1], edge[1], graph, layernumbers - i - 1, [], [])
            # print('pt1', pt1)
            # print('pt2', pt2)
            if pt2 != [] or pt1 != []:
                for i1 in pt1:
                    for j1 in pt2:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)

            # print(edge[1])
            # print(i)
            pt3 = dfs(goal, goal, edge[1], graph, i + 2, [], [])
            pt4 = dfs2(edge[0], edge[0], graph, layernumbers - i - 1, [], [])
            # print('pt3', pt3)
            # print('pt4', pt4)
            if pt3 != [] or pt4 != []:
                for i1 in pt3:
                    for j1 in pt4:
                        # print(i1 + j1)
                        if i1 + j1 not in resultpath:
                            resultpath.append(i1 + j1)
    return resultpath
def deepliftrelumultigaijin(goal,k,layernumbers,goalpaths,Hnew,Hold,addedgelist,W,X): #有relu层
    re=dict()
    atrzong=0
    # nor_old=normalizationlist(Hold[layernumbers*2-1][goal])
    # nor_new = normalizationlist(Hnew[layernumbers * 2 - 1][goal])
    # scaling=(nor_new[k]-nor_old[k])/(Hnew[layernumbers * 2 - 1][goal][k]-Hold[layernumbers * 2 - 1][goal][k])

    for path in goalpaths:
        R=dict()
        index = layernumbers + 2
        for edge in addedgelist:
            for i in range(0, len(path) - 1):
                if path[i] == edge[0] and path[i + 1] == edge[1]:
                    if i < index:
                        index = i
                if path[i] == edge[1] and path[i + 1] == edge[0]:
                    if i < index:
                        index = i
        # print(path, index)
        if index==0:
            a = np.zeros((1, Hnew[layernumbers*2 - 2].shape[1]))
            for i in range(0, Hnew[layernumbers*2 - 2].shape[1]):
                a[0][i] = W[layernumbers][i][k]

            R[layernumbers - 1] = a
            # print(R)
            for i in range(layernumbers - 1, 0, -1):
                a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                row = 0

                for j in range(0, R[i].shape[1]):
                    x = W[i][:, j]
                    if Hnew[2 * i - 1][path[layernumbers - i]][j] != 0:
                        y = Hnew[2*i][path[layernumbers - i]][j] * x / Hnew[2 * i - 1][path[layernumbers - i]][j]
                        for m in range(0, W[i].shape[0]):
                            a[row][m] = y[m]
                            # print(a[row])
                    row = row + 1
                # print(a)
                R[i - 1] = a

        if index != 0:
            a = np.zeros((1, Hnew[layernumbers*2 - 2].shape[1]))
            for i in range(0, Hnew[layernumbers*2 - 2].shape[1]):
                # print(Hnew[layernumbers - 1].shape[1])
                a[0][i] = W[layernumbers][i][k]

            R[layernumbers - 1] = a
            # print('chazhi', (Hnew[layernumbers * 2 - 2][path[1]] - Hold[layernumbers * 2 - 2][path[1]]))
            # print((Hnew[layernumbers * 2 - 2][path[1]] - Hold[layernumbers * 2 - 2][path[1]]) * \
            #       W[layernumbers][:, k])
            for i in range(layernumbers - 1, 0, -1):
                # print(path, index)
                # print('i', i)
                if i<=(layernumbers-1 )and i >(layernumbers-index):
                    a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x = W[i][:, j]
                        if (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                            Hold[2 * i - 1][path[layernumbers - i]][
                                j]) != 0:
                            y = (Hnew[2*i][path[layernumbers - i]][j] - Hold[2*i][path[layernumbers - i]][j]) *x / (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                                     Hold[2 * i - 1][path[layernumbers - i]][
                                         j])
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]

                                # print(a[row])
                        row = row + 1
                    R[i - 1] = a
                if i ==(layernumbers - index) :
                    a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x =  W[i][:, j]
                        if (Hnew[2 * i - 1][path[layernumbers - i]][j] -
                            Hold[2 * i - 1][path[layernumbers - i]][
                                j]) != 0:
                            y = x / (
                                    Hnew[2 * i - 1][path[layernumbers - i]][j] - Hold[2 * i - 1][
                                path[layernumbers - i]][j])*(
                                    Hnew[2 * i][path[layernumbers - i]][j] - Hold[2 * i][
                                path[layernumbers - i]][j])
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]

                            # print(a[row])
                        row = row + 1

                    R[i - 1] = a
                if i <(layernumbers - index) and i >0:
                    a = np.zeros((R[i].shape[1], W[i-1].shape[1]))
                    row = 0

                    for j in range(0, R[i].shape[1]):
                        x = W[i][:, j]
                        # print('x.shape',x.shape)
                        if Hnew[2 * i - 1][path[layernumbers - i]][j] == 0:
                            a[row] = 0
                        else:
                            y = x / Hnew[2 * i - 1][path[layernumbers - i]][j]*Hnew[2 * i][path[layernumbers - i]][j]
                            # print(y.shape)
                            # print('w.SHAPE',W[1].shape)
                            for m in range(0, W[i].shape[0]):
                                a[row][m] = y[m]
                                # print(a[row])
                        row = row + 1

                    # print(a)
                    R[i - 1] = a
        # print('r0.shape',R[0].shape)
        # print(R[1].shape)
        # print(R[2].shape)
        R[0]=R[0]*X[path[-1]]
        mul = dict()
        mul[layernumbers - 2] = np.zeros((R[layernumbers - 2].shape[0], R[layernumbers - 2].shape[1]))
        for i in range(0, R[layernumbers - 1].shape[1]):
            if R[layernumbers - 1][0][i] != 0:
                mul[layernumbers - 2][i] = R[layernumbers - 1][0][i] * R[layernumbers - 2][i]
        for i in range(layernumbers - 2, 0, -1):
            mul[i - 1] = np.zeros((mul[i].shape[0] * mul[i].shape[1], R[i - 1].shape[1]))
            for j in range(0, mul[i].shape[0]):
                if (np.all(mul[i][j] == 0) == False):
                    for l in range(0, mul[i].shape[1]):
                        if mul[i][j][l] != 0:
                            mul[i - 1][j * mul[i].shape[1] + l] = mul[i][j][l] * R[i - 1][l]
        attr = mul[0]

        atr = attr.sum(0)
        atr =atr.sum(0)
        # print('atr',atr.shape)
        # # print('y_min', y_min)
        # # print('y_max',y_max)
        # print(Hnew[0].shape[1])
        # print('atr',atr)
        atr=atr
        # print('atr', atr)

        # print('R',R)
        # print(R[0].sum(0))
        strpath = []
        for pathindex in path:
            strpath.append(str(pathindex))
        c = ','.join(strpath)
        re[c] = atr
        # if path==[0,0,4,0] or path==[0,0,3,0]:
        #     print(path,R)
        # print(R[0].sum(1).sum(0))
        atrzong = atrzong + atr.sum(0)
    return re, atrzong
def main_con(number1,number2,length,goal,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hnew[layernumbers*2-1][goal] )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==a+Hold[layernumbers*2-1][goal]]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_graph(number1,number2,length,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hnew )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(-d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==a+Hold]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_mask(number1,number2,length,goal,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hold[layernumbers*2-1][goal] )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==Hnew[layernumbers*2-1][goal]-a]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='SCS') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_con_mask_graph(number1,number2,length,ma,num_classes,Hnew,Hold,layernumbers): #convex

    x = cvx.Variable(ma.shape[0])
    y=cvx.Variable(ma.shape[1])

    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    #
    # c=a+Hold[layernumbers*2-1][goal]
    # print('c.shape',c.shape)
    # print("constraints is DCP:", (sum(x) == length).is_dcp())
    d=0
    for i in range(0,ma.shape[1]):
        d=d+a[i]*softmax(Hold )[i]
    # e=0
    # for i in range(0,ma.shape[1]):
    #     # e=e+cvx.atoms.exp(c[i])
    #     e = e + cvx.atoms.exp(y[i])
    # print('e',e)
    # print('e.shape',e.shape)
    #
    #
    #
    #
    # # f=sum(c*softmax(Hnew[layernumbers*2-1][goal]))
    # # print(c*softmax(Hnew[layernumbers*2-1][goal]).shape)
    # # print(cvx.atoms.log_sum_exp(c).shape)
    objective = cvx.Minimize(d+cvx.atoms.log_sum_exp(y))
    constraints = [sum(x)== length, y==Hnew-a]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEKMOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def edge_index_both(edges_dict,pa_add,pa_remove,edges_new):
    remove_path_1=[]
    remove_index_1=[]
    remove_path_2=[]
    remove_index_2=[]
    add_path_1=[]
    add_path_2=[]
    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            # print('it',it)
            # if (path[1], path[2]) not in remove_path_1:
            #     remove_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in remove_path_1:
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2:
                add_path_2.append((path[1], path[0]))
                remove_index_2.append(edges_dict[(path[1], path[0])])
    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2:
                add_path_2.append((path[1], path[0]))
            if (path[1], path[0]) not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
                remove_path_2.append((path[1], path[0]))
                remove_index_2.append(edges_dict[(path[1], path[0])])
    remove_index_1=list(set(remove_index_1))
    remove_index_1 = sorted(remove_index_1)
    remove_index_2 = list(set(remove_index_2))
    remove_index_2 = sorted(remove_index_2)

    edges_1 = copy.deepcopy(edges_new)  # h

    # print('add_path_1', add_path_1)
    # print('add_path_2',add_path_2)
    # print('remove_path_1',remove_path_1)
    # print('remove_path_2', remove_path_2)

    for i in reversed(remove_index_1):
        # print((edges_1[0][i],edges_1[1][i]))
        del edges_1[0][i]
        del edges_1[1][i]
    for path in add_path_1:
        edges_1[0].append(path[0])
        edges_1[1].append(path[1])
    edges_2 = [[], []]  # adj2
    for path in add_path_2:
        edges_2[0].append(path[0])
        edges_2[1].append(path[1])
    edges_3 = copy.deepcopy(edges_new)  # adj1
    # print('remove_index_2',remove_index_2)
    for j in reversed(remove_index_2):
        # print((edges_3[0][j], edges_3[1][j]))
        del edges_3[0][j]
        del edges_3[1][j]
    # print('add_path_1',add_path_1)
    # print('add_path_2', add_path_2)
    edges_index_1 = torch.tensor(edges_1)
    edges_index_2 = torch.tensor(edges_2)
    edges_index_3 = torch.tensor(edges_3)

    return edges_index_1, edges_index_2, edges_index_3

def XAIxiugai(goal,k,layernumbers,goalpaths,H,W):  #有relu层的多层GCN
    # h1=matrix.dot(X).dot(W1)
    # h3=matrix.dot(h1).dot(W2)
    epsilon = 1e-30
    #
    re=dict()
    atrzong=0
    for path in goalpaths:
        # print(path)
        R = dict()
        a = np.zeros(H[layernumbers * 2 - 2].shape[1])
        a = H[layernumbers * 2 - 2][path[1]] * W[layernumbers][:, k] / (H[layernumbers * 2 - 1][goal][k] +epsilon)* \
            H[layernumbers * 2 - 1][goal][k]
        R[layernumbers - 1] = a.reshape((1, a.shape[0]))

        # if H[layernumbers*2-1][goal][k] != 0:
        #     a = H[layernumbers*2 - 2][path[1]] * W[layernumbers][:, k] / H[layernumbers*2-1][goal][k] * H[layernumbers*2-1][goal][k]
        #     # print('shape1',H[layernumbers*2 - 2][path[1]].shape)
        #     # print('shape2',W[layernumbers][:, k].shape)
        #     R[layernumbers - 1] = a.reshape((1, a.shape[0]))
        # else:
        #     a = np.zeros(H[layernumbers*2 - 2].shape[1])
        #     R[layernumbers - 1] = a.reshape((1, a.shape[0]))

        for i in range(layernumbers - 1, 0, -1):
            # print(i)
            a = np.zeros((R[i].shape[1] * R[i].shape[0], W[i - 1].shape[1]))
            row = 0
            for l in range(0, R[i].shape[0]):
                for j in range(0, R[i].shape[1]):
                    a[row] = H[2 * i - 2][path[layernumbers + 1 - i]] * W[i][:, j] / \
                             (H[2 * i - 1][path[layernumbers - i]][j]+epsilon) * R[i][l][j]
                    row=row+1
                    # if H[2*i-1][path[layernumbers - i]][j] == 0:
                    #     a[row] = np.zeros(W[i - 1].shape[1])
                    #     row = row + 1
                    # else:
                    #     # print('1', H[2*i - 2][path[layernumbers + 1 - i]])
                    #     # print('2', W[i][:, j])
                    #     # print('3', R[i][l][j])
                    #     # print('4',a[row])
                    #     a[row] = H[2*i - 2][path[layernumbers + 1 - i]] * W[i][:, j] / H[2*i-1][path[layernumbers - i]][j] * R[i][l][j]
                    #     # print(a[row])
                    #     row = row + 1
            # print(a)
            R[i - 1] = a
            # print(R)
        atr = R[0].sum(1).sum(0)
        atrzong = atr + atrzong
        # print( atr, path)
        # print(R)
        strpath = []
        for index in path:
            strpath.append(str(index))
        c = ','.join(strpath)
        re[c] = atr
    return re,atrzong
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)
    # print('node_idx',node_idx)
    # print('node_mask',node_mask)
    # print('edge_mask', edge_mask)
    # print(len(row))
    inv = None

    subsets = [node_idx]

    for _ in range(num_hops):
        # print(num_hops)
        node_mask.fill_(False)
        # print(node_mask)
        node_mask[subsets[-1]] = True
        # print(row)
        torch.index_select(node_mask, 0, row, out=edge_mask)
        # print(torch.index_select(node_mask, 0, row, out=edge_mask))
        subsets.append(col[edge_mask])
    # print(subsets)
    # print('lensubsets',len(subsets[3]))
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    # print('inv',inv)
    # print('subset',subset)
    inv = inv[:node_idx.numel()]
    # print(subsets)
    # print('inv', inv)
    # print(node_idx.numel())

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    # print(edge_mask)
    # print(len(edge_mask))

    edge_index = edge_index[:, edge_mask]
    # print(len(edge_index[0]))
    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
def map_edges(subedges,mapping,features,nodelist):
    map_edges=[[],[]]
    sub_array = []
    sub_dict=dict()
    subfeatures=np.zeros((len(nodelist),features.shape[1]))
    for idx,node in enumerate(subedges[0]):
        map_edges[0].append(mapping[node])
        map_edges[1].append(mapping[subedges[1][idx]])
        sub_array.append((mapping[node], mapping[subedges[1][idx]]))
        sub_dict[(mapping[node], mapping[subedges[1][idx]])]=idx
    for key,value in mapping.items():
        subfeatures[mapping[key]]=features[key]

    sub_array = np.array(sub_array)
    # print('sub_array', sub_array)
    adj = sp.coo_matrix((np.ones(sub_array.shape[0]), (sub_array[:, 0], sub_array[:, 1])),
                        shape=(len(nodelist), len(nodelist)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return map_edges,subfeatures,adj,sub_dict
def forward_tensor(adj,layernumbers,W): #有relu
    hiddenmatrix = dict()
    # adj = torch.tensor(adj, requires_grad=True)
    # adj=sparse_mx_to_torch_sparse_tensor(adj)
    relu=torch.nn.ReLU(inplace=False)
    hiddenmatrix[0] = W[0]


    h = torch.sparse.mm(adj, W[0])

    hiddenmatrix[1] = torch.mm(h, W[1])
    hiddenmatrix[2]=relu(hiddenmatrix[1])
    # hiddenmatrix[1].retain_grad()
    for i in range(1, layernumbers):
        h = torch.sparse.mm(adj, hiddenmatrix[2*i])
        hiddenmatrix[2*i + 1] = torch.mm(h, W[i + 1])
        if i!=layernumbers-1:
            hiddenmatrix[2*i+2]=relu(hiddenmatrix[2*i + 1])
        # hiddenmatrix[i + 1].retain_grad()
    return hiddenmatrix
def main_linear(number,length,goal,ma,num_classes,Hnew,layernumbers):

    x = cvx.Variable(ma.shape[0]) #boolean=True


    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    d = 0
    for i in range(0, ma.shape[1]):
        d = d + a[i] * softmax(Hnew[layernumbers * 2 - 1][goal])[i]



    objective = cvx.Minimize(number-d)
    constraints = [sum(x)== length]
    for i in range(0, ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
        # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK')

    # for i in range(0,ma.shape[0]):
    #     constraints.append(0 <= x[i])
    #     constraints.append(x[i] <= 1)
    # print(constraints)

    #
    # prob.solve(solver=cvx.CPLEX)
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print(x.value)  # A numpy ndarray.**
    return x.value
def main_linear_graph(number,length,ma,num_classes,Hnew,layernumbers):

    x = cvx.Variable(ma.shape[0]) #boolean=True


    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    d = 0
    for i in range(0, ma.shape[1]):
        d = d + a[i] * softmax(Hnew)[i]



    objective = cvx.Minimize(number-d)
    constraints = [sum(x)== length]
    for i in range(0, ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
        # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK')

    # for i in range(0,ma.shape[0]):
    #     constraints.append(0 <= x[i])
    #     constraints.append(x[i] <= 1)
    # print(constraints)

    #
    # prob.solve(solver=cvx.CPLEX)
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print(x.value)  # A numpy ndarray.**
    return x.value
def edge_index_add(edges_dict,pa,edges_old):

    add_path_1 = []
    # add_index_1 = []
    remove_path_2 = []
    remove_index_2 = []
    add_path_2 = []

    for it, path in enumerate(pa):
        # print('it',it)
        # if (path[1], path[2]) not in add_path_1:
        #     add_path_1.append((path[1], path[2]))
        if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
            add_path_1.append((path[2], path[1]))
            # add_index_1.append(edges_dict[(path[2], path[1])])
        if (path[1], path[0]) not in add_path_2:
            add_path_2.append((path[1], path[0]))
        if (path[1], path[0])  not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
            remove_path_2.append((path[1], path[0]))
            remove_index_2.append(edges_dict[(path[1], path[0])])


    remove_index_2 = sorted(remove_index_2)
    # print(add_path_1)
    # print(add_path_2)

    # delindex=0
    addedges_1 = copy.deepcopy(edges_old) #h
    # print('add_index_1',add_index_1)
    for path in add_path_1:
        addedges_1[0].append(path[0])
        addedges_1[1].append(path[1])

    deledges_2=[[],[]]  #adj2
    for path in add_path_2:
        deledges_2[0].append(path[0])
        deledges_2[1].append(path[1])
    # print('deledges_2',deledges_2)


    deledges_3 = copy.deepcopy(edges_old)  #adj1
    # print('remove_index_2',remove_index_2)
    for j in reversed(remove_index_2):
        # print((deledges_3[0][j], deledges_3[1][j]))
        del deledges_3[0][j]
        del deledges_3[1][j]

    deledges_index_1 = torch.tensor(addedges_1)
    deledges_index_2 = torch.tensor(deledges_2)
    deledges_index_3 = torch.tensor(deledges_3)

    return deledges_index_1,deledges_index_2,deledges_index_3
def smooth(arr, eps=1e-5):
    if 0 in arr:
        return abs(arr - eps)
    else:
        return arr


def KL_divergence(P, Q):
    # Input P and Q would be vector (like messages or priors)
    # Will calculate the KL-divergence D-KL(P || Q) = sum~i ( P(i) * log(Q(i)/P(i)) )
    # Refer to Wikipedia https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    P = smooth(P)
    Q = smooth(Q)
    return sum(P * np.log(P / Q))
def rumor_construct_adj_matrix(edges_index,x):
    edges = []
    # print(ground_truth.keys())
    for idx,node in enumerate(edges_index[0]):
        edges.append((node,edges_index[1][idx]))


    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(x, x),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj
def clear_time(time_dict):
    edge_time = dict()
    for key, value in time_dict.items():
        month = (value.year - 2010) * 12 + value.month
        week = (value.year - 2010) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
def clear_time_UCI(time_dict):
    edge_time = dict()
    for key, value in time_dict.items():
        month = (value.year - 2004) * 12 + value.month
        week = (value.year - 2004) * 52 + value.isocalendar()[1]
        edge_time[key]=(value.year,month,week)
    clear_time = dict()
    for key, value in edge_time.items():
        if (key[1], key[0]) in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
        if (key[1], key[0]) not in edge_time.keys():
            clear_time[key] = value
            clear_time[(key[1], key[0])] = value
    return clear_time
def split_edge(start,end,flag,clear_time):
    edge_index = [[], []]
    if flag == 'year':
        for key, value in clear_time.items():
            if value[0] >= start and value[0] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    if flag == 'month':
        for key, value in clear_time.items():
            if value[1] >= start and value[1] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])

    if flag=='week':
        for key, value in clear_time.items():
            if value[2] >= start and value[2] < end:
                edge_index[0].append(key[0])
                edge_index[1].append(key[1])
    return edge_index
def subadj_map(subset,edge_index):
    mapping = dict()
    for it, neighbor in enumerate(subset):
        mapping[neighbor.item()] = it
    # print(mapping)
    # print(np.array(edge_index))
    con_edges = []
    for idx, edge in enumerate(edge_index[0]):
        con_edges.append((edge.item(), edge_index[1][idx].item()))
    edges = np.array(con_edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(subset), len(subset)),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return mapping,adj
def subH(subset,mapping,H,layernumbers):
    sub_H=dict()


    for i in range(0,layernumbers*2):
        subarray=np.array(np.zeros((len(subset),H[i].shape[1])))
        for j in range(len(subset)):
            subarray[mapping[subset[j].item()]]=H[i][subset[j]]
        sub_H[i]=subarray
    return sub_H

def subh1(subset,mapping,W,layernumbers):
    h1=W[0].dot(W[1])
    subarray = np.zeros((len(subset), h1.shape[1]))
    for j in range(len(subset)):
        subarray[mapping[subset[j].item()]] = h1[subset[j]]
    return subarray
def subpath_edge(goalnewaddpath,addedgelist,submapping,edge_index):
    goalnewaddpathmap = []
    for it, path in enumerate(goalnewaddpath):
        pathmap = []
        for j in range(0, len(path)):
            pathmap.append(submapping[path[j]])
        goalnewaddpathmap.append(pathmap)
    # print(goalnewaddpathmap)
    # print(goalnewaddpath)
    addedgelistmap = []
    for it, edge in enumerate(addedgelist):
        # edge_map=[]
        # print(edge[0])
        # print(edge[1])
        if edge[0] in submapping.keys() and edge[1] in submapping.keys():
            addedgelistmap.append([submapping[edge[0]], submapping[edge[1]]])
        # if edge_map!=[]:
        #     addedgelistmap.append(edge_map)
        # if edge[0] in submapping.keys() and edge[1] in submapping.keys():
        #     edge_map.append([submapping[edge[0]],submapping[edge[1]]])
    # print(addedgelistmap)
    subedgesmap = []


    for idx, edge in enumerate(edge_index[0]):
        # print(edge[0])
        # print(edge[1])
        # print(edge[idx])
        subedgesmap.append((edge.item(), edge_index[1][idx].item()))

    return goalnewaddpathmap,addedgelistmap,subedgesmap
def subpath_goalpath(resultdict,submapping):
    submapping_revse=dict((value,key) for key,value in submapping.items())
    resultdict_goalpath=dict()
    for key,value in resultdict.items():
        deepliftpath = []
        goalpath=[]
        s1 = key.split(',')
        for j in s1:
            deepliftpath.append(submapping_revse[int(j)])
        for pathindex in deepliftpath:
           goalpath.append(str(pathindex))
        c = ','.join(goalpath)
        resultdict_goalpath[c] = value
    return resultdict_goalpath
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def edge_index_both_g0(edges_dict,pa_add,pa_remove,edges_old,removeedgelist): #g0出发增加重要路径
    add_path_1 = []
    add_path_2 = []
    remove_path_1 = []
    remove_index_1 = []
    remove_path_2 = []
    remove_index_2 = []
    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            # print('it',it)
            # if (path[1], path[2]) not in add_path_1:
            #     add_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2 and (path[1], path[0]) not in edges_dict.keys():
                add_path_2.append((path[1], path[0]))
            # if (path[1], path[0])  not in remove_path_2 and (path[1], path[0]) in edges_dict.keys():
            #     remove_path_2.append((path[1], path[0]))
            #     remove_index_2.append(edges_dict[(path[1], path[0])])
        if removeedgelist!=[]:
            for remve_path in removeedgelist:
                for it,path in enumerate(add_path_1):
                    node=path[1]
                    if node==remve_path[0]:
                        if (remve_path[1],node) not in remove_path_1:
                            remove_path_1.append((remve_path[1],node))
                            remove_index_1.append(edges_dict[(remve_path[1],node)])
                    elif node==remve_path[1]:
                        if ( remve_path[0],node) not in remove_path_1:
                            remove_path_1.append((remve_path[0],node))
                            remove_index_1.append(edges_dict[(remve_path[0],node)])






    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            # print('it',it)
            # if (path[1], path[2]) not in remove_path_1:
            #     remove_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in remove_path_1 and (path[2], path[1]) in edges_dict.keys():
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
            # if (path[0], path[1]) not in remove_path_2:
    # print('add_path_1',add_path_1)
    # print('add_path_2',add_path_2)
    # print('remove_path_1',remove_path_1)

    remove_index_1 = sorted(remove_index_1)

    both_edges_1 = copy.deepcopy(edges_old)  # h
    # print('remove_index_1',remove_index_1)
    for i in reversed(remove_index_1):
        # print((deledges_1[0][i],deledges_1[1][i]))
        del both_edges_1[0][i]
        del both_edges_1[1][i]


    for path in add_path_1:
        both_edges_1[0].append(path[0])
        both_edges_1[1].append(path[1])

    addedges_2 = copy.deepcopy(edges_old)  # h
    for path in add_path_2:
        addedges_2[0].append(path[0])
        addedges_2[1].append(path[1])
    deledges_index_1 = torch.tensor(both_edges_1)
    deledges_index_2 = torch.tensor(addedges_2)
    # deledges_index_3 = torch.tensor(deledges_3)

    return deledges_index_1, deledges_index_2
def edge_index_both_g1(edges_dict,pa_add,pa_remove,edges_new,addedgelist):
    remove_path_1 = []
    remove_index_1 = []
    add_path_1=[]
    add_path_2=[]


    if pa_add!=[]:
        for it, path in enumerate(pa_add):
            if (path[2], path[1]) not in remove_path_1:
                remove_path_1.append((path[2], path[1]))
                remove_index_1.append(edges_dict[(path[2], path[1])])
    if pa_remove!=[]:
        for it, path in enumerate(pa_remove):
            # print('it',it)
            # if (path[1], path[2]) not in add_path_1:
            #     add_path_1.append((path[1], path[2]))
            if (path[2], path[1]) not in edges_dict.keys() and (path[2], path[1]) not in add_path_1:
                add_path_1.append((path[2], path[1]))
                # add_index_1.append(edges_dict[(path[2], path[1])])
            if (path[1], path[0]) not in add_path_2 and (path[1], path[0]) not in edges_dict.keys():
                add_path_2.append((path[1], path[0]))
        if addedgelist!=[]:
            for add_path in addedgelist:
                for it, path in enumerate(add_path_1):
                    node = path[1]
                    if node == add_path[0]:
                        if (add_path[1], node) not in remove_path_1:
                            remove_path_1.append((add_path[1], node))
                            remove_index_1.append(edges_dict[(add_path[1], node)])
                    elif node == add_path[1]:
                        if (add_path[0], node) not in remove_path_1:
                            remove_path_1.append((add_path[0], node))
                            remove_index_1.append(edges_dict[(add_path[0], node)])

    remove_index_1 = sorted(remove_index_1)

    both_edges_1 = copy.deepcopy(edges_new)  # h
    # print('remove_index_1',remove_index_1)
    for i in reversed(remove_index_1):
        # print((deledges_1[0][i],deledges_1[1][i]))
        del both_edges_1[0][i]
        del both_edges_1[1][i]

    for path in add_path_1:
        both_edges_1[0].append(path[0])
        both_edges_1[1].append(path[1])

    addedges_2 = copy.deepcopy(edges_new)  # h
    for path in add_path_2:
        addedges_2[0].append(path[0])
        addedges_2[1].append(path[1])
    deledges_index_1 = torch.tensor(both_edges_1)
    deledges_index_2 = torch.tensor(addedges_2)
    # deledges_index_3 = torch.tensor(deledges_3)

    return deledges_index_1, deledges_index_2