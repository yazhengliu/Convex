def time_step(dataset,t):
    if dataset=='Chi':
        if t!='both':
            return [[84,90],[90,96],[96,102],[102,108]]
        else:
            return [[84,80],[86,82],[88,84],[90,86]]
    elif dataset=='NYC':
        if t!='both':
            return [[78,80],[80,82],[82,84],[84,86]]
        else:
            return [[84,79],[85,80],[86,81],[87,82]]
    elif dataset=='Zip':
        if t!='both':
            return [[78,79],[79,80],[80,81],[81,82]]
        else:
            return [[354,340],[356,342],[358,344],[360,346]]
    elif dataset=='pheme' or dataset=='weibo':
        if t=='add':
            return[['edges_1','edges_2'],['edges_2','edges_3']]
        elif t=='remove':
            return [['edges_2', 'edges_1'], ['edges_3', 'edges_2']]
        else:
            return [['edges_2', 'edges_4']]
    elif dataset=='bitcoinalpha':
        if t=='add' or t=='remove':
            return [[48,51],[51,54],[54,57],[57,60]]
        else:
            return [[48,50],[50,52],[52,54],[54,56]]
    elif dataset=='bitcoinotc':
        if t=='add' or t=='remove':
            return [[48,50],[52,54],[54,56],[56,58]]
        else:
            return [[48,50],[50,52],[52,54],[54,56]]
    elif dataset=='UCI':
        if t=='add' or t=='remove':
            return [[18,19],[19,20],[20,21],[21,22]]
        else:
            return [[19,20],[20,21],[21,22],[22,23]]




def path_number(dataset,t,targetpath):
    if dataset=='Chi' or dataset=='NYC' or dataset=='Zip':
        if t=='add' or t=='both' or t=='remove':
            # return [1,2,3,4,5]
            #
            if targetpath > 1000:
                return [15, 16, 17, 18, 19]
            elif 500 < targetpath <= 1000:
                return [10, 11, 12, 13, 14]
            elif 100 < targetpath < 500:
                return [6, 7, 8, 9, 10]
            else:
                return [1,2,3,4,5]

    elif dataset=='weibo' or dataset=='pheme':
        if t == 'add' or t == 'both' or t == 'remove':
            if targetpath > 1000:
                return [15, 16, 17, 18, 19]
            elif 500 < targetpath <= 1000:
                return [10, 11, 12, 13, 14]
            elif 100 < targetpath < 500:
                return [6, 7, 8, 9, 10]
            else:
                return [1, 2, 3, 4, 5]
            # if targetpath > 1000:
            #     return [60,70,80,90,100]
            # elif 500 < targetpath <= 1000:
            #     return [10, 20, 30, 40, 50]
            # elif 100 < targetpath < 500:
            #     return [10,15,20,25,30]
            # else:
            #     return [1, 2, 3, 4, 5]
    elif dataset=='bitcoinalpha' or dataset=='bitcoinotc' or dataset=='UCI':
        if t == 'add' or t == 'both' or t == 'remove':
            if targetpath > 1000:
                return [60,70,80,90,100]
            elif 500 < targetpath <= 1000:
                return [10,20,30,40,50]
            elif 100 < targetpath < 500:
                return [10,12,14,16,18]
            else:
                return [1, 2, 3, 4, 5]

    elif dataset=='mutag':
        if t == 'add' or t == 'both' or t == 'remove':
            # return [1,2,3,4,5]
            if targetpath > 1000:
                return [10,11,12,13,14]
            elif 500 < targetpath <= 1000:
                return [6,7,8,9,10]
            elif 100 < targetpath < 500:
                return [3,4,5,6,7]
            else:
                return [1, 2, 3, 4, 5]




