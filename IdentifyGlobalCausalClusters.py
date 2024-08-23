import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import GINTest
import Utils
from itertools import combinations


warnings.filterwarnings("ignore")

UseRank = 0
Limitedfactor = 3



# Test GIN condition by auto selecting methods
def Test_GIN_Condition(Y, Z, data, alpha = 0.01, Grlen = 2):
    Y=list(Y)
    Z=list(Z)

    if len(Y) > len(Z):
        m = GINTest.GINTest(Y, Z, data, alpha, 'fast')

    else:
        try:
            m = GINTest.GINTest(Y, Z, data, alpha, 'rank', Grlen-1)
        except ValueError:
            print('==========exog is collinear Error! =============> ', Y, Z)
            m = GINTest.GINTest(Y, Z, data, alpha, 'fast')

    return m




def IdentifyGlobalCausalClusters(data ,LatentIndex ,Ora_data, alpha = 0.005):

    """
    Identify global causal clusters from an active variable set A.

    Parameters:
    data: observation data (pd.Dataframe)
    LatentIndex: Privous learned causal cluster
    Ora_data: the original data that can be used in the extended method if set 'UseRank' to 1.


    Returns:
    ClusterList: A dictionary where keys are latent variables, and values are lists of identified clusters
    """

    indexs = list(data.columns)
    A = indexs.copy()
    Cluster = {}
    B = A.copy()
    Grlen = 2

    #can not be test by GIN condition
    if len(indexs) < 3:
        print('There are something wrong when perform the identify global causal cluster! The current dataset is: ', indexs)
        return Cluster

    while (len(B) >= Grlen and len(indexs) >= 2*Grlen-1) and Grlen <= Limitedfactor:
        Set_P = FindCombination(B, Grlen)

        for P in Set_P:
            tind = indexs.copy()
            for t in P:
                tind.remove(t)  #   tind= ALLdata\P

            # Finding Causal Cluster by Rank constraints if the dimsion is hopefull large
            if UseRank:
                print("The original Test target are : ", P, tind)
                tdata = data.copy()
                P1,tind1,tdata = Extend_Observed_Dim(P, tind, tdata, Ora_data, LatentIndex)
                print('Extend Observed Dim as :',P1, tind1)
            else:
                tdata = data.copy()
                P1 = P
                tind1 = tind

            if Grlen <= 2:
                if Test_GIN_Condition(P1, tind1, tdata, alpha, Grlen):
                    key = Cluster.keys()
                    key = list(key)
                    if 1 in key:
                        temp = Cluster[1]
                        temp.append(list(P))
                        Cluster[1] = temp
                    else:
                        Cluster[1] = [list(P)]
            else: # complete impure detecting

                ##try to use rank, if len(Z) > len(X) > (Grlen-1)
                if len(P1) <= len(tind1):
                    latentNum = GetClusterbyRank(P1,tind1,tdata, alpha)
                else: # no enough to use rank test!
                    latentNum = GetCluster(P1,tind1,tdata, alpha)

                if latentNum == -1:
                    continue
                if latentNum <= (Grlen-1): #latent
                    key = Cluster.keys()
                    key = list(key)
                    if latentNum in key:
                        temp = Cluster[latentNum]
                        temp.append(list(P))
                        Cluster[latentNum] = temp

                    else:
                        Cluster[latentNum] = [list(P)]

        print('Current iteration learning causal cluster are : ', Cluster)



        TempList=Getlist(Cluster)

        B = [item for item in B if item not in TempList]

        Grlen+=1
        #print(B)
        print()

    return Cluster



#get all observed in X and Z
def Extend_Observed_Dim(P1, tind1, tdata, Ora_data, LatentIndex):

    if len(LatentIndex) == 0:
        return P1, tind1, tdata

    P = list(P1)
    tind = list(tind1)
    X = []
    Z = []
    key = list(LatentIndex.keys())
    C1 = recursiveFindObserved(P,LatentIndex)
    C2 = recursiveFindObserved(tind,LatentIndex)

    for i in C1:
        if i not in X:
            X.append(i)
            index = list(tdata.columns)
            if i not in index:
                tdata[i] = Ora_data[i]

    for i in C2:
        if i not in Z:
            Z.append(i)
            index = list(tdata.columns)
            if i not in index:
                tdata[i] = Ora_data[i]

    for i in X:
        if i in Z:
            Z.remove(i)

    return list(X),list(Z),tdata


def recursiveFindObserved(C,LatentIndex):
    X=set()
    key = list(LatentIndex.keys())
    for i in C:
        if i in key:
            clu = LatentIndex[i]
            C2 = recursiveFindObserved(clu,LatentIndex)
            for j in C2:
                X.add(j)

        else:
            X.add(i)

    return list(X)


def Getlist(r):
    a=[]
    key = r.keys()
    if len(key) ==0:
        return []
    for i in key:
        t = r[i]
        for j in t:
            for n in j:
                a.append(n)

    return a


def GetCluster(group, tind, data, alpha):
    lens = len(group)
    for i in range(2, lens+1):
        Set = FindCombination(group, i)
        flag = True
        for j in Set:
            if not Test_GIN_Condition(list(j), list(tind), data, alpha):
                flag = False
                break

        if flag:
            return i-1


    return -1


def GetClusterbyRank(group, tind, data, alpha = 0.01):
    lens=len(group)
    for i in range(2, lens+1):
        if Test_GIN_Condition(group, tind, data, alpha, i):
            return i-1

    return -1


def FindCombination(Lists,N):
    return itertools.combinations(Lists,N)


