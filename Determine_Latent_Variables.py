import numpy as np
import pandas as pd
import itertools
import networkx as nx
import warnings
import Utils
import GINTest
import Identify_Pure_Cluster

NotAllowNfactorMerge = 1
debug = 1


def MergerCluster(Cluster,data,Ora_data,LatentIndex,alpha = 0.01):
    """
    Determine Latent Variables based on the given cluster set, active variable set, and partial graph.

    Parameters:

    Cluster               : A list of clusters
    data                  : Active variable set
    LatentIndex           : Partial graph

    Returns:
    LatentIndex           : Updated partial graph with determined latent variables
    """


    Cluster = MergingOverlapCluster(Cluster)



    PureCluster, TempImpureClu = GetPureCluster(Cluster, data, LatentIndex, Ora_data, alpha)

    if debug:
        print('The PureCluster are identified as: ', PureCluster)

    # if there not any new cluter is learned
    if not Cluster:
        return Cluster, LatentIndex, data, PureCluster

    key = list(Cluster.keys())
    if debug:
        print('++++++++Stage I: Identify Proposition 4 Rule 1++++++++++++++')

    for LatentNum in key:

        if NotAllowNfactorMerge and LatentNum >=2:
            break

        CluList = Cluster[LatentNum].copy()

        #Mergering two cluster when there are the same latent parent

        for C1 in CluList:
            MergeSet = []
            if not CheckPure(C1,LatentNum, PureCluster):
                TCluList = CluList.copy()
                TCluList.remove(C1)
                for C2 in TCluList:
                    if CheckMergeRule1(C1, C2, LatentNum, LatentNum, PureCluster, data, LatentIndex, alpha):
                        MergeSet.append(C2)
            #merge and update the cluster
            if len(MergeSet) >0:
                for c in MergeSet:
                    CluList.remove(c)
                C1index = CluList.index(C1)
                CluList.remove(C1)
                NewC = C1
                for c in MergeSet:
                    NewC = set(NewC).union(set(c))
                CluList.insert(C1index,list(NewC))
                Cluster[LatentNum] = CluList

                #update the pure cluster
                PureClu = [C1[0]] + [C2[0] for C2 in MergeSet]
                PureCluster[LatentNum].append(PureClu)



    if debug:
        print('++++++++Stage II: Identify Proposition 4 (Rule 2)++++++++++++++')

    for i in range(len(key)-1, 0, -1):
        for j in range(0, i-1):
            List1=Cluster[key[i]]
            List2=Cluster[key[j]]
            for C1 in List1: #biger atoms
                MergeSet = []
                for C2 in List2: #smaller atom
                    LC1=int(key[i])
                    LC2=int(key[j])
                    if CheckMergeRule2(C1, C2, LC1, LC2, Cluster, PureCluster, data, LatentIndex, Ora_data, alpha):
                        MergeSet.append(C2)
                #merge and update the cluster
                if len(MergeSet) >0:
                    for c in MergeSet:
                        List2.remove(c)
                    C1index = CluList.index(C1)
                    List1.remove(C1)
                    NewC = C1
                    for c in MergeSet:
                        NewC = set(NewC).union(set(c))
                    List1.insert(C1index,list(NewC))
                    Cluster[key[i]] = List1
                    Cluster[key[j]] = List2

                    #update the pure cluster
                    PureClu = [C1[0]] + [C2[0] for C2 in MergeSet]
                    PureCluster[LatentNum].append(PureClu)



    if debug:
        print('++++++++Stage III: Identify Corollary 1 R3 ++++++++++++++')

    if len(list(LatentIndex.keys())) == 0:
        return Cluster, LatentIndex, data, PureCluster


    for LatentNum in key:

        if NotAllowNfactorMerge and LatentNum >=2:
            break
        LatentNames=list(LatentIndex.keys())
        CluList = Cluster[LatentNum].copy()
        for C1 in CluList:
            for L in C1:
                if L in LatentNames and not CheckNfactor(L,LatentIndex):
                    if CheckMergerRule3(L, C1, 1, LatentNum, PureCluster, data, LatentIndex, Ora_data, alpha):
                        if debug:
                            print('~~~~~~~~~~~~>> Merger Rule hold for Rule 3: ', L, C1)
                        #Step I: remove the cluster C1 from Cluster
                        TC1 = C1.copy()
                        TC1.remove(L)
                        CluList.remove(C1)
                        Cluster[LatentNum] =CluList
                        #Step II: Update the LatentIndex (update the causal structure)
                        TClu = LatentIndex[L]
                        for i in TC1:
                            TClu.append(i)
                        LatentIndex[L] = TClu
                        #Step III: Update the current dataset
                        indexs = [col for col in list(data.columns) if col not in TC1]
                        data = data[indexs]

                        #Step IV: Update the pureCluster set for each latent variable, if need.
                        PureClu = [TC1[0], LatentIndex[L][0]]
                        PureCluster[LatentNum].append(PureClu)

                        #Step V: Stop checking the early learning
                        break
        if debug:
            print('++++++++Stage III_sub-stage 2: Identify Corollary 1 R3 for the different latent number ++++++++++++++')


        #if there exist the biger atomix merger
        for C1 in CluList:
            C1_candinate = GetNfactorLatents(C1, LatentIndex)
            for Lp in C1_candinate:
                TC1 = [var for var in C1 if var not in Lp]
                if CheckMergerRule5(Lp, TC1, len(Lp), LatentNum, PureCluster, data, LatentIndex, Ora_data, alpha):
                    if len(TC1) == 0:
                        continue
                    if debug:
                        print('~~~~~~~~~~~~>> Merger Rule hold for Rule 5: ', Lp, C1)
                     #Step I: remove the cluster C1 from Cluster
                    CluList.remove(C1)
                    Cluster[LatentNum] =CluList
                     #Step II: Update the LatentIndex (update the causal structure)
                    TClu = LatentIndex[Lp[0]]
                    TC1 = [var for var in C1 if var not in Lp]
                    for i in Lp:
                        LatentIndex[i] = list(set(TClu).union(set(TC1)))
                     #Step III: Update the current dataset
                    indexs = [col for col in data.columns if col not in TC1]
                    data = data[indexs]
                    break





    if debug:
        print('++++++++Stage IV: Identify Corollary 1 R4 ++++++++++++++')

    A=list(data.columns)
    ClusterLists=Getlist(Cluster)
    B=set()
    for i in A:
        if (i not in ClusterLists) and (i in list(LatentIndex.keys())):
            B.add(i)
    B = list(B)

    for LatentNum in key:
        if NotAllowNfactorMerge and LatentNum >=2:
            break
        LatentNames=list(LatentIndex.keys())
        CluList = Cluster[LatentNum].copy()
        for C1 in CluList:
            for L in B:
                if CheckNfactor(L,LatentIndex):
                    continue
                if CheckMergeRule4(L, C1, 1, LatentNum, PureCluster, data, LatentIndex, Ora_data, alpha):
                    if debug:
                        print('~~~~~~~~~~~~>> Merger Rule hold for Rule 4: ', L, C1)
                    #Step I: remove the cluster C1 from Cluster
                    CluList.remove(C1)
                    Cluster[LatentNum] =CluList
                    #Step II: Update the LatentIndex (update the causal structure)
                    TClu = LatentIndex[L]
                    for i in C1:
                        TClu.append(i)
                    LatentIndex[L] = TClu
                    #Step III: Update the current dataset
                    indexs = [col for col in data.columns if col not in C1]
                    data = data[indexs]

                    #Step IV: Update the pureCluster set for each latent variable, if need.
                    PureClu = [C1[0], LatentIndex[L][0]]
                    PureCluster[LatentNum].append(PureClu)
                    #Step V: Stop checking the early learning
                    break

        if debug:
            print('++++++++Stage IV-substage 2: Identify Corollary 1 R4 for the different latent number ++++++++++++++')

        #if there exist the biger atomix merger
        for C1 in CluList:
            C2_candinate = GetNfactorLatents(B, LatentIndex)
            for Lp in C2_candinate:
                if CheckMergerRule6(Lp, C1, len(Lp), LatentNum, PureCluster, data, LatentIndex, Ora_data, alpha):
                    if debug:
                        print('~~~~~~~~~~~~>> Merger Rule hold for Rule 6: ', Lp, C1)
                    #Step I: remove the cluster C1 from Cluster
                    CluList.remove(C1)
                    Cluster[LatentNum] =CluList
                    #Step II: Update the LatentIndex (update the causal structure)
                    TClu = LatentIndex[Lp[0]]
                    for i in Lp:
                        LatentIndex[i] = list(set(TClu)+set(C1))
                    #Step III: Update the current dataset
                    indexs = [col for col in data.columns if col not in C1]
                    data = data[indexs]
                    break

    return Cluster, LatentIndex, data, PureCluster





def CheckNfactor(L,LatentIndex):
    LatentNames=list(LatentIndex.keys())
    for i in range(0, len(LatentNames)):
        if i == len(LatentNames)-1:
            return False
        L2 = LatentNames[i]
        L3 = LatentNames[i+1]
        if L == L2:
            Set1 = LatentIndex[L2]
            Set2 = LatentIndex[L3]

            if set(Set1) == set(Set2):
                return True

            if i > 0:
                L4 = LatentNames[i-1]
                Set3 = LatentIndex[L4]
                if set(Set1) == set(Set3):
                    return True

    return False





def CheckMergeRule1(C1, C2, LC1, LC2, PureCluster, data, LatentIndex,  alpha = 0.01):
    BugDetector(data, C1, C2, LC1)

    A = list(data.columns)
    LatentNum = LC1
    for item in C1 + C2:
        if item in A:
            A.remove(item)
    Y=[]
    Z=A.copy()
    if CheckPure(C2, LC2, PureCluster):
        Y.append(C1[0])
        Y.extend(C2[:LatentNum])
        Z.extend(C2[LatentNum:])
##        if GINTest.GINTest(Y, Z, data, alpha, "fast"):
##            return True

    else:
        Y = [C1[0], C2[0]]

    if debug:
        print('Check MergerRule 1----------> ', C1, C2, Y, Z)

    if GINTest.GINTest(Y, Z, data, alpha, "fast"):
        return True

    return False



def CheckMergeRule2(C1, C2, LC1, LC2, Cluster, PureCluster, data, LatentIndex, Ora_data, alpha = 0.01):
    BugDetector(data, C1, C2, LC1)

    A = list(data.columns)
    for item in C1 + C2:
        if item in A:
            A.remove(item)

    if CheckPure(C1, LC1, PureCluster) and CheckPure(C2, LC2, PureCluster):
        return False

    #LC1 > LC2
    Y = []
    Z = A.copy()
    #Case I: C1 is pure and C2 is impure
    if CheckPure(C1, LC1, PureCluster) and not CheckPure(C2, LC2, PureCluster):
        Y.extend(C1[:LC1])
        Z.extend(C1[LC1:])
        Y.append(C2[0])

    #Case II: C2 is pure and C1 is impure
    elif CheckPure(C2, LC2, PureCluster) and not CheckPure(C1, LC1, PureCluster):
        #find the sub-pure cluster from the n-factor impure cluster
        subPureC1 = Identify_Pure_Cluster.Identify_subPure_Cluste(C1, Cluster, LC1, data, LatentIndex, Ora_data, alpha)
        if len(subPureC1) == 0:
            Y.extend(C1[:LC1])
            Y.append(C2[0])
            Z.extend(C2[1:])
        else:
            Y.append(C2[0])
            Y.append(subPureC1[0])
            Z.append(subPureC1[1])
            TC1 = C1.copy()
            TC1.remove(subPureC1[0])
            TC1.remove(subPureC1[1])
            Z.extend(TC1[LC1-1:])
            Y.extend(TC1[:LC1-1])

    #Case III: C2 and C1 both are impure
    elif not CheckPure(C2, LC2, PureCluster) and not CheckPure(C1, LC1, PureCluster):
        #find the sub-pure cluster from the n-factor impure cluster
        subPureC1 = Identify_Pure_Cluster.Identify_subPure_Cluste(C1, Cluster, LC1, data, LatentIndex, Ora_data, alpha)
        if len(subPureC1) == 0:
            Y.extend(C1[:LC1])
            Y.append(C2[0])
        else:
            Y.append(subPureC1[0])
            Z.append(subPureC1[1])
            TC1 = C1.copy()
            TC1.remove(subPureC1[0])
            TC1.remove(subPureC1[1])
            Y.extend(TC1[:LC1-1])
            Y.append(C2[0])

    if len(Y) <= 1:
        return False

    if debug:
        print('Check MergerRule 2----------> ', C1, C2, Y, Z)
    if GINTest.GINTest(Y, Z, data, alpha, "fast"):
        return True

    return False




def CheckMergerRule3(L, C1, LC2, LC1, PureCluster, data, LatentIndex, Ora_data, alpha = 0.01):
    #Note that L is in the C1
    TC1 = C1.copy()
    if L in TC1:
        TC1.remove(L)
    #extend the data, called 'tdata'
    PureDes, tdata = GetPureDescendants(L, LatentIndex, data, Ora_data)
    C2 = list(PureDes)

    A_tilde = [col for col in tdata.columns if col not in C1+C2]

    Z = A_tilde.copy()
    Y = []

    Z.append(C2[0])
    Y.append(C2[1])
    Y.extend(TC1)

    if debug:
        print('Check MergerRule 3----------> ', C1, C2, Y, Z)

    if GINTest.GINTest(Y, Z, tdata, alpha, "fast"):
        return True

    return False




def CheckMergeRule4(L, C1, LC2, LC1, PureCluster, data, LatentIndex, Ora_data, alpha = 0.01):

    if LC1 == LC2:
        return CheckMergerRule3(L, C1, LC2, LC1, PureCluster, data, LatentIndex, Ora_data, alpha)

    PureDes, tdata = GetPureDescendants(L, LatentIndex, data, Ora_data)

    C2 = list(PureDes)

    A_tilde = [col for col in tdata.columns if col not in C1+C2]


    Z = A_tilde.copy()
    Y = []

    subPureC1 = Identify_Pure_Cluster.Identify_subPure_Cluste(C1, Cluster, LatentNum, data, LatentIndex, Ora_data, alpha)

    if len(subPureC1) == 0:
        Y.extend(C1[:LC1])
        Y.append(C2[0])
        Z.append(C2[1])
    else:
        Y.append(subPureC1[0])
        Z.append(subPureC1[1])
        Y.append(C2[0])
        Z.append(C2[1])
        C_tilde = [var for var in C1 if var not in subPureC1]
        Y.extend(C_tilde)

    if debug:
        print('Check MergerRule 4----------> ', C1, C2, Y, Z)

    if GINTest.GINTest(Y, Z, tdata, alpha, "fast"):
        return True

    return False


#Note that Lp is a list for the case Lp sub to C1
def CheckMergerRule5(Lp, C2, LC1, LC2, PureCluster, data, LatentIndex, Ora_data, alpha):

    if LC1 <= LC2: #this procedure is checked in the Rule 3
        return False

    PureDes, tdata = GetPureDescendants2(Lp, LatentIndex, data, Ora_data)

    C1_tilde = PureDes[:LC1]

    for V_i in C2:
        Y = C1_tilde.copy()
        Z = [var for var in list(tdata.columns) if var not in Lp+C2+[V_i]]
        Y.append(V_i)
        Z.extend(C1_tilde[LC1:])
        if not GINTest.GINTest(Y, Z, tdata, alpha, "fast"):
            return False

    return True





#Note that Lp is a list
def CheckMergerRule6(Lp, C2, LC1, LC2, PureCluster, data, LatentIndex, Ora_data, alpha):

    if LC1 == LC2 and LC1 == 1:
        return False

    PureDes, tdata = GetPureDescendants2(Lp, LatentIndex, data, Ora_data)
    C1_tilde = PureDes[:LC1]

    for V_i in C2:
        Y = C1_tilde.copy()
        Y.append(V_i)
        Z = [var for var in list(tdata.columns) if var not in Lp+C2+[V_i]]
        Z.extend(C1_tilde[LC1:])
        if not GINTest.GINTest(Y, Z, tdata, alpha, "fast"):
            return False

    return True





#Get 2-factor only, can be extend to get n-factor, edited by user
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



def GetPureDescendants2(Lp, LatentIndex, data, Ora_data):
    tdata = data.copy()
    LatentName = list(LatentIndex.keys())
    ObseredIndexs = list(Ora_data.columns)
    PureDes = []
    for L in Lp:
        del tdata[L]


    Clu = LatentIndex[Lp[0]]




    for i in range(0, len(Lp)+1):
        try:
            T1 = Clu[i]
            while T1 in LatentIndex and T1 not in ObseredIndexs:
                T1 = LatentIndex[T1][0]
            PureDes.append(T1)
        except IndexError:
            print('list index out of range for Clu: ', Clu, Lp)
            break

    for T1 in PureDes:
        if T1 in ObseredIndexs:
            tdata[T1] = Ora_data[T1]
        else:
            print('There are something wrong!!')
            exit(-1)

    return PureDes, tdata







def GetPureDescendants(L, LatentIndex, data, Ora_data):
    tdata = data.copy()
    LatentName = list(LatentIndex.keys())
    ObseredIndexs = list(Ora_data.columns)
    del tdata[L]
    Clu = LatentIndex[L]

    T1 = Clu[0]
    T2 = Clu[1]

    while T1 in LatentIndex and T1 not in ObseredIndexs:
        T1 = LatentIndex[T1][0]
    while T2 in LatentIndex and T2 not in ObseredIndexs:
        T2 = LatentIndex[T2][0]

    PureDes = [T1, T2]

    if T1 in ObseredIndexs and T2 in ObseredIndexs:
        tdata[T1] = Ora_data[T1]
        tdata[T2] = Ora_data[T2]
    else:
        print('There are something wrong!!')
        exit(-1)

    return PureDes, tdata







def CheckPure(C1,LatentNum, PureCluster):

    ClusterList = PureCluster[LatentNum]

    for clu in ClusterList:
        if set(clu) == set(C1):
            return True

    return False



def MergingOverlapCluster(Cluster):
    key = list(Cluster.keys())

    for Llen in key:
        Clu = Cluster[Llen]
        Clu = Utils.merge_lists_overlap(Clu)
        Cluster[Llen] = Clu


    for i in range(len(key)-1, 0, -1):
        List1=Cluster[key[i]]
        for C1 in List1:
            for j in range(0, i-1):
                List2=Cluster[key[j]]
                MergeSet = []
                for C2 in List2:
                    if set(C1) & set (C2):
                        MergeSet.append(C2)
                if len(MergeSet) >0:
                    MergeSet.insert(0, C1)
                    TC1 = Utils.merge_lists_overlap(MergeSet)
                    C1index = List1.index(C1)
                    List1.remove(C1)
                    List1.insert(C1index,TCl)
                    for c in MergeSet:
                        List2.remove(c)
                    Cluster[key[i]] = List1
                    Cluster[key[j]] = List2
    return Cluster


def GetPureCluster(Cluster, data, LatentIndex, Ora_data, alpha = 0.01):
    PureCluster = {}
    ImpureCluster = {}

    key = Cluster.keys()
    for Llen in key:
        Clu = Cluster[Llen]
        TempPureClu = []
        TempImpureClu = []
        for C in Clu:

            Impureflag = Identify_Pure_Cluster.Identify_Pure_Causal_Cluster(C, Cluster, Llen, data, LatentIndex, Ora_data, alpha)
            if not Impureflag:
                TempPureClu.append(C)
            else:
                TempImpureClu.append(C)
        PureCluster[Llen] = TempPureClu
        ImpureCluster[Llen] = TempImpureClu


    return PureCluster, TempImpureClu



#transfer dic to list
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


def BugDetector(data, C1, C2, LC1):
    if not isinstance(data, pd.DataFrame):
        print('Please ensure the data is the type of pd.DataFrame!', type(data))
        exit(-1)
    if not (isinstance(C1, list) and len(C1) >= 2):
        print('Please ensure C1 is a list with at least two elements!', C1)
        exit(-1)
    if not (isinstance(C2, list) and len(C2) >= 2):
        print('Please ensure C1 is a list with at least two elements!', C2)
        exit(-1)
    if not ((isinstance(LC1, int)) or (isinstance(LC1, str) and LC1.isdigit())):
        print('Please ensure LC1 is a list with at least two elements!', LC1)
        exit(-1)





