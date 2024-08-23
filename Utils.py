import numpy as np
import pandas as pd
from itertools import combinations
import itertools
import networkx as nx

#从latent index 从返回2-factor的情况
def Get_n_Factor(LatentIndex):
    key=LatentIndex.keys()
    key=list(key)
    Latent=[]

    record=[]

    for i in key:
        if i in record:
            continue
        clu = LatentIndex[i]
        tl=[]
        tl.append(i)
        record.append(i)
        for j in key:
            if i == j:
                continue
            tclu =LatentIndex[j]
            if clu == tclu:
                tl.append(j)
                record.append(j)
        Latent.append(tl)
##    print(Latent)
    return Latent

#test : {'L1':}
def GetLatentIndex(latentName,LatentIndex):
    Latent = Get_n_Factor(LatentIndex)
    for i in Latent:
        if latentName in i:
            t = i
            break

    Index = t.index(latentName)
    key=LatentIndex.keys()
    key=list(key)
    tname =latentName
    while tname in key:
        tname = LatentIndex[tname][Index]

    return tname









#Overleaf的合并_version 1
def merge_list_1(L2,lens):

    #print(L2)

    L1=[]
    Temp=[]
    for i in L2:
        if len(i) != int(lens+1):
            Temp.append(i)
            continue
        else:
            L1.append(i)

    #print(L1)



    l = L1.copy()
    G = nx.Graph()

    G.add_nodes_from(sum(l, []))

    q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in l]
    for i in q:

        G.add_edges_from(i)

    aa =[list(i) for i in nx.connected_components(G)]

    if len(Temp) >0:  #非纯加在后面
        for i in Temp:
            aa.append(i)

    return aa


def bfs(graph, start):
    visited, queue = set(), [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

def connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            c = set(bfs(G, v))
            yield c
            seen.update(c)

def graph(edge_list):
    result = {}
    for source, target in edge_list:
        result.setdefault(source, set()).add(target)
        result.setdefault(target, set()).add(source)
    return result

#Overleaf的合并_version 1
def merge_list_old(L2,lens):


    L1=[]
    Temp=[]
    for i in L2:
        if len(i) != int(lens+1):
            Temp.append(i)
            continue
        else:
            L1.append(i)
    l=L1.copy()

    edges = []
    s = list(map(set, l))
    for i, j in combinations(range(len(s)), r=2):
        if s[i].intersection(s[j]):
            edges.append((i, j))
    G = graph(edges)
    result = []
    unassigned = list(range(len(s)))
    for component in connected_components(G):
        union = set().union(*(s[i] for i in component))
        result.append(sorted(union))
        unassigned = [i for i in unassigned if i not in component]
    result.extend(map(sorted, (s[i] for i in unassigned)))

    if len(Temp) >0:  #非纯加在后面
        for i in Temp:
            result.append(i)
    return result


#Overleaf的合并_version 1
def merge_list(L2):



    l=L2.copy()

    edges = []
    s = list(map(set, l))
    for i, j in combinations(range(len(s)), r=2):

        if s[i].intersection(s[j]):
            edges.append((i, j))

    G = graph(edges)

    result = []
    unassigned = list(range(len(s)))

    for component in connected_components(G):

        union = set().union(*(s[i] for i in component))

        result.append(sorted(union))

        unassigned = [i for i in unassigned if i not in component]


    result.extend(map(sorted, (s[i] for i in unassigned)))

    return result

#update the cluster with overlap, untile without any new cluster merge
#C1-C6-->H1-H3->H4->NULL




#尽管能返回集合的共同元素，可是变成集合的时候似乎序列会边
def merge_listNew(L2):



    l=L2.copy()


    edges = []
    s = list(map(set, l))
    for i, j in combinations(range(len(s)), r=2):
        if s[i].intersection(s[j]):
            edges.append((i, j))

    G = graph(edges)

    result = []
    unassigned = list(range(len(s)))

    for component in connected_components(G):

        component=list(component)

        Cluster=l[component[0]]

        adj= s[0].intersection(s[1])
        for i in component:
            if i == 0:
                continue
            temp=l[i]
            for j in list(adj):
                if j in temp:
                    temp.remove(j)

            for j in range(0,len(temp)):
                if j < (len(temp)/2):
                    Cluster.insert(0,temp[j])
                else:
                    Cluster.append(temp[j])

        result.append(Cluster)



        unassigned = [i for i in unassigned if i not in component]


    result.extend(map(sorted, (s[i] for i in unassigned)))

    return result


def merge_lists_overlap(lists):
    merged_lists = []

    while lists:
        first, *rest = lists
        first = set(first)

        lf = -1
        while len(first) > lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if first.intersection(r):
                    first |= set(r)
                else:
                    rest2.append(r)
            rest = rest2

        merged_lists.append(list(first))
        lists = rest

    return merged_lists




def main():
    #L=[['x1','x2'],['x3','x4'],['x5','x6','x2'],['x1','x9']]
    L=[['x1','x3','x4','x5','x2'],['x6','x9','x7'],['x10','x9'],['x15','x3','x4','x5','x17']]
    print(merge_lists_overlap(L))




if __name__ == '__main__':
    main()
