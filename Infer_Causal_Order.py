import numpy as np
import pandas as pd
import GINTest

#Cluster = [[]], K is causal order, e.g., K = [['x1','x2']] where ['x1','x2'] is root
def LearnCausalOrder(Cluster, K, data):

    while(len(Cluster) >0):
        root = FindRoot(Cluster,data,K)
        K.append(root)
        Cluster.remove(root)
    return K

#finding root node by mutual information based GIN
def FindRoot(clu1,data1,K1):
    cluster=clu1.copy()
    data =data1.copy()
    K=K1.copy()
    if len(cluster) ==1:
        return cluster[0]

    MIS=[]
    for clu in cluster:
        other=cluster.copy()
        other.remove(clu)
        TempM=0
        for o in other:
            X=[]
            Z=[]

            for i in range(0,len(clu)):
                if i < len(clu)/2:
                    X.append(clu[i])
                else:
                    Z.append(clu[i])
            for i in range(0,len(o)):
                if i < len(o)/2:
                    X.append(o[i])


            if len(K) != 0:
                for i in K:
                    for j in range(0,len(i)):
                        if j < int(len(i)/2):
                            X.append(i[j])
                        else:
                            Z.append(i[j])

            #print(X,Z)
            mi = GINTest.FisherGIN_byfastHSIC(X, Z, data)
            #mi = GIN.GIN_MI(X,Z,data,method='1')
            TempM+=mi
            TempM =TempM
        MIS.append(TempM)

    mins = MIS.index(min(MIS))
    root = cluster[mins]
    return root

