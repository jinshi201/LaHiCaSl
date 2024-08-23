import numpy as np
import pandas as pd
from itertools import combinations
import itertools
import GINTest


NotAllowNfactorImpure = 1


# Test GIN condition by auto selecting methods
def Test_GIN_Condition(Y, Z, data, alpha = 0.01, Grlen = 2, SelectedMethod = "fast"):

    if SelectedMethod == "fast":
        return GINTest.GINTest(Y, Z, data, alpha, 'fast')

    Y=list(Y)
    Z=list(Z)

    if len(Y) > len(Z):
        m = GINTest.GINTest(Y, Z, data, alpha, 'fast')

    else:
        m = GINTest.GINTest(Y, Z, data, alpha, 'rank', Grlen-1)

    return m





'''
Identify Pure&Impure causal cluster according to Lemma 1

'''
def Identify_Pure_Causal_Cluster(C, Cluster, LatentNum, data, LatentIndex, Ora_data, alpha = 0.01):
    LatentName = list(LatentIndex.keys())

    #More efficient if there is some prior
    if NotAllowNfactorImpure and LatentNum >=2:
            return False


    # large than the latent factor, C must be impure, e.g., L->{X1 X2 X3}, then dim{X1 X2 X3} >2, it must be impure
    #Condition (1) of Lemma 1
    if len(C) > LatentNum+1:
        return False


    #Condition (2) of Lemma 1
    #One more effective method is to select V_k from the learned cluster, which can be edited by user
    A = list(data.columns)
    Impureflag = False
    if len(C) == LatentNum+1:
        SV_ij = FindCombination(list(C), 2)
        for V_ij in SV_ij:
            V_i = V_ij[0]
            V_j = V_ij[1]
            A_tilde = list(set(A) - set(C))

            #Find P and V_k in A_tilde
            Vk_candidate = SelectVk(C, Cluster, data, alpha, A)

            #print(Vk_candidate)

            #no V_k can be found, it must be pure cluster according to the model definition
            if len(Vk_candidate) == 0:
                return Impureflag


            for V_k in Vk_candidate:
                B = A_tilde.copy()
                if V_k in B:
                    B.remove(V_k)

                P_candidate = FindCombination(B, LatentNum)

                for P in P_candidate:
                    P = list(P)
                    Z1 = P.copy()
                    Z1.append(V_i)
                    Y = C.copy()
                    Y.append(V_k)
                    Z2= P.copy()
                    Z2.append(V_j)
                    if Test_GIN_Condition(Y, Z1, data, alpha, LatentNum) and not Test_GIN_Condition(Y, Z2, data, alpha, LatentNum):
                        print('One can see that the condition (2) hold for', Y, Z1, Z2)
                        Impureflag = True
                        break


                if Impureflag:
                    return Impureflag


    return Impureflag


#One can limite the dimension of subset. Default to 2
# if on any sub-set is pure cluster, return an empty list
def Identify_subPure_Cluste(C, Cluster, LatentNum, data, LatentIndex, Ora_data, alpha = 0.01, limitedLens = 2):


    subsets = [subset for r in range(2, limitedLens + 1) for subset in itertools.combinations(C, r)]
    A = list(data.columns)

    for C_tilde in subsets:
        C_tilde = list(C_tilde)

        #condition (1) of Lemma 2
        if len(C_tilde) >= LatentNum +1:
            Impureflag = Identify_Pure_Causal_Cluster(C_tilde, Cluster, LatentNum, data, LatentIndex, Ora_data, alpha)

            if not Impureflag: #one can replace this with the maximal pure-sub cluster, which can be edited by user
                return C_tilde


        #condition (2) of Lemma 2
        if len(C_tilde) < LatentNum +1 :
            Impureflag = False
            SV_ij = FindCombination(list(C), 2)
            for V_ij in SV_ij:
                V_i = V_ij[0]
                V_j = V_ij[1]
                A_tilde = list(set(A) - set(C))

                QLen = LatentNum +1 - len(C_tilde)

                Q_candinate = FindCombination(A_tilde, QLen)

                for Q in Q_candinate:
                    Q = list(Q)
                    B_tilde = list(set(A_tilde) - set(Q))

                    P_candinate = FindCombination(B_tilde, LatentNum)

                    for P in P_candinate:
                        P = list(P)
                        Y = C_tilde.copy()
                        Y = list(set(Y).union(set(Q)))
                        Z1 = P.copy()
                        Z1.append(V_i)

                        Z2 = P.copy()
                        Z2.append(V_j)

                        if Test_GIN_Condition(Y, Z1, data, alpha, LatentNum) and not Test_GIN_Condition(Y, Z2, data, alpha, LatentNum):
                            Impureflag = True
                            break

                    if Impureflag:
                        return C_tilde

    return []







def SelectVk(C, Clusters1, data, alpha, A):

    Clusters = GetCluster(Clusters1)

    TestVk=set()
    for V_list in Clusters:
        if set(C) == set(V_list):
            continue
        Vk = V_list[0]

        Z = set (A) - set(C)
        Z = set(Z) - set(V_list)
        Z = list(Z)


        X=list(C).copy()
        X.append(Vk)
        if len(Z) == 0:
            break
        if Test_GIN_Condition(X, Z, data, alpha):
            TestVk.add(Vk)

    return list(TestVk)





#obtain the vaild surrogate for learned latent variable
def GetIndexByObserved(LatentIndex):
    LatentName = list(LatentIndex.keys())

    result={}

    for i in LatentName:
        dex1 = LatentIndex[i][0]
        dex2 = LatentIndex[i][1]
        while dex1 in LatentName:
            dex1 = LatentIndex[dex1][0]
        while dex2 in LatentName:
            dex2 = LatentIndex[dex2][1]
        result[i] = [dex1,dex2]

    return result



# obtain all learned causal cluster set
def GetCluster(Cluster):
    key = Cluster.keys()

    Lists = []

    for i in key:
        Cs = Cluster[i]
        for j in Cs:
            Lists.append(j)
    return Lists




def FindCombination(Lists,N):
    return itertools.combinations(Lists,N)
