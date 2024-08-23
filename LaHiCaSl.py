import numpy as np
import pandas as pd
import MakeGraph
import itertools
#Phase I: Locate latent variables
import IdentifyGlobalCausalClusters  # Stage I-S1
import Determine_Latent_Variables # Stage I-S2
import UpdateActiveData # Stage I-S3
#Phase II: Infer causal structure among latent variables
import LocallyInferCausalStructure #Stage II

debug = 1


def Latent_Hierarchical_Causal_Structure_Learning(data, alpha):

    """

    Latent Hierarchical Causal Structure Learning (LaHiCaSL)

    Parameters:
    data : set of observed variables
    alpha: Threshold

    Returns:
    Causal_Matrix : Causal structure matrix over both observed and latent variables

    """

    #Initialize latent index
    L_index=1
    #Initialize the ora data
    Ora_data = data.copy()
    # Initialize the graph
    LatentIndex = {}
    # Initialize CLuster set, that recored each learning result for different iteration
    AllPureCluster = []
    AllLearnedClusters = []
    AllImpureCluster = []
    # Initialize the signficant level
    alpha = 0.05




    #Phase I: Locate latent variables
    print('Begin with Phase I: Locate latent variables +++++++++++++++++++')

    while(True):

        # Stage I-S1 ← IdentifyGlobalCausalClusters



        Cluster = IdentifyGlobalCausalClusters.IdentifyGlobalCausalClusters(data ,LatentIndex ,Ora_data, alpha)

        #if there not any new latent variable (cluster) is found
        if not Cluster or all(not v for v in Cluster.values()):
            break


        #Stage I-S2 ← DetermineLatentVariables

        Cluster, LatentIndex, data, PureCluster = Determine_Latent_Variables.MergerCluster(Cluster, data, Ora_data, LatentIndex, alpha)

        #All cluster is early learning, (that is a measurement-model)
        if not Cluster:
            break


        #Stage I-S3 ← UpdateActiveData

        LatentIndex, data, L_index = UpdateActiveData.UpdateActiveData(Cluster, L_index, LatentIndex, data, PureCluster, Ora_data)

        AllLearnedClusters, AllImpureCluster, AllPureCluster, LatentIndex = UpdateActiveData.UpdateAllClusterInformation(Cluster, PureCluster, AllLearnedClusters, AllPureCluster, AllImpureCluster, LatentIndex)

        if debug:
            print('=========> The Impure Cluster Set: ', AllImpureCluster)

    print('End of Phase I: Locate latent variables +++++++++++++++++++')


    #Phase II: Infer causal structure among latent variables


    print('Begin with Phase II: Infer causal structure among latent variables +++++++++++++++++++')

    LocallyInferCausalStructure.LocalLearningStructure(AllImpureCluster, LatentIndex, Ora_data)

    print('End of Phase II: Infer causal structure among latent variables +++++++++++++++++++')



    Causal_Matrix = MakeGraph.UpdateGraph(list(Ora_data.columns),LatentIndex)

    print('================ The result of structure learning (adj matrix) : \n', Causal_Matrix)


    #Draw a graph
    #MakeGraph.Make_graph(LatentIndex)





