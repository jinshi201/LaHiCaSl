import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import Utils
import GINTest
import Infer_Causal_Order



def LocalLearningStructure(AllImpureCluster, LatentIndex, Ora_data):
    """
    Locally Infer the Causal Structure.

    Parameters:
    Ora_data : set of observed variables
    LatentIndex : partial structure (graph)

    Returns:
    LatentIndex : Fully identified causal structure
    """

    AllImureList = GetImpureClusterList(AllImpureCluster)
    UpdateOrder = []

    for C in AllImureList:
        Order = InferOrder(C, LatentIndex, Ora_data)

    #Update LatentIndex

    LatentName = list(LatentIndex.keys())

    for order in UpdateOrder:
        for i in range(0, len(order)):
            root = order[i]
            if root not in LatentName:
                print('do not satify the model assumption!')
                return LatentIndex
            clu = LatentIndex[root]
            clu.extend(order[i+1:])
            LatentIndex[root] = clu

    return LatentIndex







def GetImpureClusterList(AllImpureCluster):
    AllImureList = []

    #Get the various level learned results
    for Level_Plists in AllImpureCluster:
        for Lnum in Level_Plists.keys():
            Plist = Level_Plists[Lnum]
            for clu in Plist:
                if clu:
                    AllImureList.append(clu)

    return AllImureList



def InferOrder(C, LatentIndex, Ora_data):
    LatentName = list(LatentIndex.keys())


    Lindex = {}
    #Find the confounder structure
    for L in LatentName:
        LClu = LatentIndex[L]
        if set(C).issubset(set(LClu)):
            Lindex[L] = LClu

    #Determine the N-factor
    if not Lindex:
        print('Cannot find C cluster in the structure! There are some wrong!')
        exit(-1)

    Parent4L = list(Lindex.keys())

    Lchildren = list(set(LClu) - set(C))

    Lchildren = Lchildren[:len(Parent4L)]

    if len(Lchildren) < 2:
        print('There are something wrong when infer the causal order: ', C, Lchildren)
        return C

    #Step I: get the pure descendant children for each latent varaible in C
    #Cluster is cluster, index is the index for each element in the cluster


    Lchildren = GetPureCluster4L(Lchildren, LatentIndex, list(Ora_data.columns))
    Cluster4C, index4C, NewC = GetPureCluster4C(C, LatentIndex, list(Ora_data.columns))


    #Step II: update the date set according to ora_data
    tdata = Ora_data[index4C + Lchildren]


    #Step III: infer the causal order, based on the measurement model approach
    K = []
    K.append(Lchildren)

    CausalOrder = Infer_Causal_Order.LearnCausalOrder(Cluster4C, K, tdata)

    #Step IV: trans order to the latent index and update the latent index

    OrderC = TranOrder2Index(CausalOrder[1:], NewC, Cluster4C, LatentIndex)

    return OrderC




def TranOrder2Index(CausalCluster, C, Cluster4C, LatentIndex):

    Order = []

    for clu in CausalCluster:
        for i in range(0, len(C)):
            C_candinate = Cluster4C[i]
            if set(clu) == set(C_candinate):
                if isinstance(C[i], list):
                    for j in C[i]:
                        Order.append(j)
                else:
                    Order.append(C[i])

    return Order



def GetPureCluster4L(Lchildren, LatentIndex, A_index):

    ObservedDescendant = []

    LatentName = list(LatentIndex.keys())

    for L in Lchildren:
        while L in LatentName and L not in A_index:
            L = LatentIndex[L][0]
        ObservedDescendant.append(L)

    return ObservedDescendant


def GetPureCluster4C(C, LatentIndex, A_index):
    ObservedDescendant4C = []
    LatentName = list(LatentIndex.keys())
    ObservedIndex = []

    Lp = GetNfactorLatents(C, LatentIndex)

    Lp_descendant = GetPureDescendants2(Lp, LatentIndex, A_index)


    ObservedIndex = Lp_descendant.copy()

    C = [var for var in C if var not in Lp]

    ObservedDescendant4C.append(Lp_descendant)

    for L in C:
        clu = GetPureDescendants(L, LatentIndex, ObseredIndexs)
        ObservedDescendant4C.append(clu)
        if set(ObservedIndex).intersection(set(clu)):
            print('there are something overlap descendant!')
            exit(-1)
        ObservedIndex = set(ObservedIndex) + set(clu)

    NewC = [Lp]
    for c in C:
        NewC.append(c)

    return ObservedDescendant4C, list(ObservedIndex), NewC




def GetNfactorLatents(C, LatentIndex):

    dic = LatentIndex.copy()
    lst = [col for col in C if col in list(LatentIndex.keys())]


    inverse_dic = {}
    for key in lst:

        value = frozenset(dic[key])
        if value not in inverse_dic:
            inverse_dic[value] = []
        inverse_dic[value].append(key)

    result = [keys for keys in inverse_dic.values() if len(keys) > 1]

    return result


def GetPureDescendants2(Lp, LatentIndex, ObseredIndexs):

    LatentName = list(LatentIndex.keys())

    PureDes = []

    Clu = LatentIndex[Lp[0]]
    for i in range(0, len(Lp)+1):
        T1 = Clu[i]
        while T1 in LatentIndex and T1 not in ObseredIndexs:
            T1 = LatentIndex[T1][0]
        PureDes.append(T1)


    return PureDes




def GetPureDescendants(L, LatentIndex, ObseredIndexs):

    LatentName = list(LatentIndex.keys())

    Clu = LatentIndex[L]
    T1 = Clu[0]
    T2 = Clu[1]

    while T1 in LatentIndex and T1 not in ObseredIndexs:
        T1 = LatentIndex[T1][0]
    while T2 in LatentIndex and T2 not in ObseredIndexs:
        T2 = LatentIndex[T2][0]

    PureDes = [T1, T2]

    return PureDes
