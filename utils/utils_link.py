import numpy as np
import torch
import cvxpy as cvx
from utils.utils_deeplift import softmax
def mlp_XAI_linear_nobas(layernumbers_mlp,H,W,hidden,nclass,index):
    a = np.zeros((1, W[1].shape[0]))
    for i in range(0, W[1].shape[0]):
      a[0][i] = W[1][i][nclass]
    return a
def XAIxiugai_link(goal,k,layernumbers,goalpaths,H,W,X,mlp_numpy,flag,hidden,k_max):  #有relu层的多层GCN
    # h1=matrix.dot(X).dot(W1)
    # h3=matrix.dot(h1).dot(W2)
    epsilon = 1e-30
    #
    re=dict()
    atrzong=0
    for path in goalpaths:
        # print(path)
        R = dict()
        a = np.zeros((1, H[layernumbers*2 - 2].shape[1]))
        for i in range(0, H[layernumbers * 2 - 2].shape[1]):
            # print(Hnew[layernumbers - 1].shape[1])
            a[0][i] = W[layernumbers][i][k]

        R[layernumbers - 1] = a
        for i in range(layernumbers - 1, 0, -1):
            a = np.zeros((R[i].shape[1], W[i - 1].shape[1]))
            row = 0

            for j in range(0, R[i].shape[1]):
                x = W[i][:, j]
                if H[2 * i - 1][path[layernumbers - i]][j] != 0:
                    y = H[2 * i][path[layernumbers - i]][j] * x / H[2 * i - 1][path[layernumbers - i]][j]
                    for m in range(0, W[i].shape[0]):
                        a[row][m] = y[m]
                        # print(a[row])
                row = row + 1
            # print(a)
            R[i - 1] = a
        R[0] = R[0] * X[path[-1]]
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
        atr = atr.sum(0)
        # print('atr',atr.shape)
        # # print('y_min', y_min)
        # # print('y_max',y_max)
        # print(Hnew[0].shape[1])
        # print('atr',atr)
        if flag == 'start':
            atr = atr * mlp_numpy[k]
        if flag == 'end':
            atr = atr * mlp_numpy[k + hidden]
        # print('atr', atr)

        # print('R',R)
        # print(R[0].sum(0))
        strpath = []
        for pathindex in path:
            strpath.append(str(pathindex))
        c = ','.join(strpath)
        re[c] = atr
        atrzong = atrzong + atr.sum(0)
        for nclass in range(1, k_max):
            a = np.zeros((1, H[layernumbers * 2 - 2].shape[1]))
            for i in range(0, H[layernumbers * 2 - 2].shape[1]):
                # print(Hnew[layernumbers - 1].shape[1])
                a[0][i] = W[layernumbers][i][nclass]

            R[layernumbers - 1] = a
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
            atr = atr.sum(0)
            if flag == 'start':
                atr = atr * mlp_numpy[nclass]
            if flag == 'end':
                atr = atr * mlp_numpy[nclass + hidden]
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] = re[c] + atr
            atrzong = atrzong + atr.sum(0)
    return re,atrzong
def forward_tensor_link(adj,layernumbers,W): #有relu
    hiddenmatrix = dict()
    # adj = torch.tensor(adj, requires_grad=True)
    # adj=sparse_mx_to_torch_sparse_tensor(adj)
    relu=torch.nn.ReLU(inplace=False)
    hiddenmatrix[0] = W[0]


    h = torch.mm(adj, W[0])

    hiddenmatrix[1] = torch.mm(h, W[1])
    hiddenmatrix[2]=relu(hiddenmatrix[1])
    # hiddenmatrix[1].retain_grad()
    for i in range(1, layernumbers):
        h = torch.mm(adj, hiddenmatrix[2*i])
        hiddenmatrix[2*i + 1] = torch.mm(h, W[i + 1])
        if i!=layernumbers-1:
            hiddenmatrix[2*i+2]=relu(hiddenmatrix[2*i + 1])
        # hiddenmatrix[i + 1].retain_grad()
    return hiddenmatrix
def grad(H_tensor,pathlist,goal,k,adj_tensor,layernumbers):
    re=dict()

    # H_tensor=H_tensor.cuda()
    H_tensor[layernumbers*2-1][goal][k].backward(retain_graph=True)
    adj_grad=adj_tensor.grad
    # print('adj_grad',adj_grad)
    adj_grad=adj_grad.numpy()
    # print('adj_grad', adj_grad)
    for path in pathlist:
        grad_attr=0
        for i in range(0,len(path)-1):
            grad_attr=grad_attr+adj_grad[path[i]][path[i+1]]
        strpath = []
        for pathindex in path:
            strpath.append(str(pathindex))
        c = ','.join(strpath)
        re[c] = grad_attr
    return re
def mlp_linear(layernumbers_mlp,Hold,Hnew,W,hidden,nclass,index_old,index_new):

    a = np.zeros((1, W[1].shape[0]))
    for i in range(0,W[1].shape[0]):
        # print('i',i)
        a[0][i] = W[1][i][nclass]

    return a
def deepliftrelumultigaijin_link(goal,k,layernumbers,goalpaths,Hnew,Hold,addedgelist,W,X,k_max,mlp_numpy,flag,hidden): #有relu层
    re=dict()
    atrzong=0
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
        if flag=='start':
            atr=atr*mlp_numpy[k]
        if flag=='end':
            atr=atr*mlp_numpy[k+hidden]
        # print('atr', atr)

        # print('R',R)
        # print(R[0].sum(0))
        strpath = []
        for pathindex in path:
            strpath.append(str(pathindex))
        c = ','.join(strpath)
        re[c] = atr
        atrzong = atrzong + atr.sum(0)
        for nclass in range(1,k_max):
            a = np.zeros((1, Hnew[layernumbers * 2 - 2].shape[1]))
            for i in range(0, Hnew[layernumbers * 2 - 2].shape[1]):
                # print(Hnew[layernumbers - 1].shape[1])
                a[0][i] = W[layernumbers][i][nclass]

            R[layernumbers - 1] = a
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
            atr = atr.sum(0)
            if flag == 'start':
                atr = atr * mlp_numpy[nclass]
            if flag == 'end':
                atr = atr * mlp_numpy[nclass + hidden]
            strpath = []
            for pathindex in path:
                strpath.append(str(pathindex))
            c = ','.join(strpath)
            re[c] =re[c]+atr
            atrzong = atrzong + atr.sum(0)
    return re, atrzong
def main_con_linear(number1,number2,length,index_new,index_old,ma,num_classes,Hnew,Hold,layernumbers): #convex

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
        d=d+a[i]*softmax(Hnew[1][index_new])[i]
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
    constraints = [sum(x)== length, y==a+Hold[1][index_old]]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value
def main_linear_mask(number1,number2,length,index_new,index_old,ma,num_classes,Hnew,Hold,layernumbers): #convex

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
        d=d+a[i]*softmax(Hold[1][index_new])[i]
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
    constraints = [sum(x)== length, y==Hnew[1][index_old]-a]

    for i in range(0,ma.shape[0]):
        constraints.append(0 <= x[i])
        constraints.append(x[i] <= 1)
    # print(constraints)
    prob = cvx.Problem(objective, constraints)
    #
    prob.solve(solver='MOSEK') #solver='SCS' CVXOPT MOSEK
    # print("Optimal value", prob.solve())
    # print("Optimal var")
    # print('x.value',x.value)  # A numpy ndarray.**
    return x.value


def main_linear_linear(number,length,index_new,ma,num_classes,Hnew,layernumbers):

    x = cvx.Variable(ma.shape[0]) #boolean=True


    # y=cvx.Variable(ma.shape[0])
    # print('x',x)
    a=np.zeros((1,num_classes))
    for i in range(0,ma.shape[0]):
        a=a+ma[i]*x[i]
    d = 0
    for i in range(0, ma.shape[1]):
        d = d + a[i] * softmax(Hnew[1][index_new])[i]



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