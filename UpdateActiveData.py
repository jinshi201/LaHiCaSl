import numpy as np
import pandas as pd
import itertools
import Utils
import warnings
warnings.filterwarnings("ignore")

debug = 1
Pure = 0



def UpdateAllClusterInformation(Cluster, PureCluster, AllLearnedClusters, AllPureCluster, AllImpureCluster, LatentIndex):

    AllLearnedClusters = UpdateAllLearnedCluster(Cluster,AllLearnedClusters)

    AllImpureCluster = UpdateAllImpureCluster(PureCluster, Cluster, AllImpureCluster)

    AllPureCluster = UpdateAllPureCluster(PureCluster, AllPureCluster)


    if Pure:
        LatentIndex = UpdatePureChildren4LatentIndex(LatentIndex, AllPureCluster, AllLearnedClusters)
        print('Complete the Update of LatentIndex, as follows: ', LatentIndex)

    return AllLearnedClusters, AllImpureCluster, AllPureCluster, LatentIndex





def UpdateActiveData(Cluster, L_index, LatentIndex, data, PureCluster, Ora_data):

    """
    Update the active variable set A based on the current state of the partial graph G.

    Parameters:
    Cluster:  Causal cluster learned in the previous procedure
    L_index: Latent index
    Ora_data : set of observed variables
    data : current active variable set
    LatentIndex : partial graph

    Returns:
    updataData : Updated active variable set
    LatentIndex : partial graph
    """

    #if not any  new cluster is learned
    if not Cluster:
        return LatentIndex, data, L_index
    Latent_Observed = []
    A = list(data.columns)
    for LatentNum in Cluster.keys():
        CluList = Cluster[LatentNum]
        for Clu in CluList:
            for i in range(0, LatentNum):
                str1='L'+str(L_index)
                L_index += 1
                Latent_Observed.append([str1,Clu[i]])

                #Update LatentIndex
                LatentIndex[str1] = Clu

            for k in Clu: #A\S
                if k in A:
                    A.remove(k)



    updataData = data[A]

    try:
        for pairIndex in Latent_Observed:
            if pairIndex[1] in list(data.columns):
                updataData[pairIndex[0]] = data[pairIndex[1]]
            else:
                K = pairIndex[1]
                while K in list(LatentIndex.keys()):
                    K = LatentIndex[K][0]
                updataData[pairIndex[0]] = Ora_data[K]


    except:
        print('There are something wrong when update the dataset! ', Latent_Observed)
        exit(-1)



    if debug:
        print('----------Update the Latent variable to actived dataset: ', list(updataData.columns))
        print('----------Update the causal graph (LatentIndex) as : ', LatentIndex)



    return LatentIndex, updataData, L_index



def UpdatePureChildren4LatentIndex(LatentIndex, AllPureCluster, AllLearnedClusters):
    if debug:
        print('I would like to update the latentIndex!', LatentIndex)
    LatentName = LatentIndex.keys()

    for L in LatentName:
        Clu = LatentIndex[L]
        #match the pure children for Clu
        PureLists = GetPureList(AllPureCluster)
        UpdateFlag = False
        for PClu in PureLists:
            Common_element = set(Clu).intersection(PClu)
            if len(Common_element) > 2:
                NClu = list(Common_element) + [x for x in Clu if x not in Common_element]
                LatentIndex[L] = NClu
                UpdateFlag = True
                break

        if not UpdateFlag:

            intersection_results = []
            list_of_sets = GetAllCluster(AllLearnedClusters)
            set1 = set(Clu)

            # 遍历 list_of_sets，检查与 set1 的交集
            for other_set in list_of_sets:
                common_elements = set1.intersection(other_set)
                if len(common_elements) >= 2:
                    intersection_results.append(list(common_elements))
            if len(intersection_results) >=2:
                TCluster = []
                for i in range(0, len(intersection_results)):
                    TCluster.append(intersection_results[i][0])
                TC = [var for var in Clu if var not in Clu]
                TCluster.extend(TC)
                LatentIndex[L] = TCluster


    return LatentIndex


def GetAllCluster(AllLearnedClusters):
    AllClusters = []

    for level in AllLearnedClusters:
        for Lnum in level.keys():
            Clu_lists = level[Lnum]
            for clu in Clu_lists:
                AllClusters.append(clu)

    return AllClusters



def GetPureList(AllPureCluster):

    AllPureList = []

    #Get the various level learned results
    for Level_Plists in AllPureCluster:
        # read the n-factor cluster
        for Lnum in Level_Plists.keys():

            Plist = Level_Plists[Lnum]
            # read the cluster in the Lnum-factor results
            for clu in Plist:
                AllPureList.append(clu)

    return AllPureCluster



def UpdateAllLearnedCluster(Cluster,AllLearnedClusters):
    if Cluster:
        AllLearnedClusters.append(Cluster)
    return AllLearnedClusters



def UpdateAllPureCluster(PureCluster, AllPureCluster):
    if PureCluster:
        AllPureCluster.append(PureCluster)
    return AllPureCluster



#Update Impure Cluster, used for local causal discovery procedure
def UpdateAllImpureCluster(PureCluster, Cluster, AllImpureCluster):
    ImpureCluster = {}

    for LatentNum, LearnCluster in Cluster.items():
        Plist = PureCluster.get(LatentNum, [])
        TempPure = {frozenset(clu) for clu in LearnCluster if frozenset(clu) in {frozenset(pclu) for pclu in Plist}}
        Impure = [clu for clu in LearnCluster if frozenset(clu) not in TempPure]
        ImpureCluster[LatentNum] = Impure

    if any(ImpureCluster.values()):
        AllImpureCluster.append(ImpureCluster)

    return AllImpureCluster





