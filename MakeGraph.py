import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt







def UpdateGraph(obserd,LatentIndex):
    key =LatentIndex.keys()

    Variables=[]
    for i in obserd:
        Variables.append(i)
    for i in key:
        if i not in Variables:
            Variables.append(i)


    n=len(Variables)
    indexs=Variables


    matrix=pd.DataFrame(np.zeros((n,n),dtype=np.int32))

    matrix.columns=indexs
    matrix.index=indexs

    for i in key:
        clu=LatentIndex[i]
        for j in clu:
            matrix[j][i]=1

    return matrix




#Plot a graph!
def Make_graph(LatentIndex):
    clusters = []
    #g = nx.empty_graph()
    g = nx.DiGraph()

    latent_nodes =list(LatentIndex.keys())
    A=[]

    for i in latent_nodes:
        g.add_node(i)
        A.append(i)
        Clu=LatentIndex[i]
        for j in Clu:
            if j not in A:
                g.add_node(j)

    for i in latent_nodes:
        Clu=LatentIndex[i]
        for j in Clu:
            g.add_edge(i, j)


    A = nx.nx_agraph.to_agraph(g)
    A.layout("dot")
    A.draw("test_tetrad.png")
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    nx.draw_networkx(g, pos, with_labels=True)
    plt.show()
