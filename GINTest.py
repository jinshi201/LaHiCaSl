import numpy as np
import pandas as pd
from indTest import *
import indTest.HSIC as hsic
import FisherTest as FT
import indTest.HSIC2 as fasthsic
import CCARankTester



def GINTest(X, Z, data, alpha = 0.01, Selectedmethod = "rank", LatentNum = 1):
    """
    Test the GIN Condition for vector X and Z.

    Parameters:
    X, Z: observed variable set
    data : dataset
    alpha : confidence interval

    Returns:
    boolean : return True if the GIN condition hold
    """
    indexs = list(data.columns)
    if X[0] not in indexs or Z[0] not in indexs:
        print('Please ensure the inpure is the variable of data!', X,Z)
        exit(-1)

    if Selectedmethod == "rank":
        return GINbyRank(X, Z, data, LatentNum, alpha)
    elif Selectedmethod == "fast":
        return GIN(X, Z, data, alpha)
    elif Selectedmethod == "kernel":
        return GIN_kernel_HSIC(X, Z, data, alpha)
    elif Selectedmethod == "fisher":
        return FisherGIN_byfastHSIC(X, Z, data, alpha)




def GINbyRank(X, Z, data, LatentNum, alpha = 0.005):

    RankTest = CCARankTester.CCARankTester_Pandas(data, alpha)

    chokePoint = LatentNum
    Flag = RankTest.test(X, Z, chokePoint)


    if Flag:
        return False
    else:
        return True




#default to fast GIN
def GIN(X,Z,data, alpha = 0.01):
    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)

    for i in Z:
        temp = np.array(data[i])
        flag =fasthsic.test(result.T,temp, alpha)

        if not flag:
            return False

    return True

def GIN_kernel_HSIC(X,Z,data, alpha = 0.01):
    #print(X,Z)

    omega = getomega(data,X,Z)
    tdata= data[X]
    #print(tdata.T)
    result = np.dot(omega, tdata.T)


    for i in Z:
        temp = np.array(data[i])
        flag = hsic.independent(result.T, temp, alpha)
        if not flag:
            return False
    return True



#fisher methods-->HSIC wiout any pval
def GIN_Fisher(X,Z,data,alph = 0.01):
    print(X,Z)

    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)

    Stat=[]
    Pvals=[]

    for i in Z:

        temp = np.array(data[i])
        pval = hsic.independent(result.T,temp)
        Pvals.append(pval)


    boolean,fpval = FT.FisherTest(Pvals,alph)
    #print(fpval)

    return boolean


def FisherGIN_byfastHSIC(X, Z, data, alph = 0.01):
    omega = getomega(data,X,Z)
    tdata= data[X]
    result = np.dot(omega, tdata.T)
    pvals=[]

    for i in Z:
        temp = np.array(data[i])
        pval=fasthsic.INtest(result.T,temp)
        pvals.append(pval)
    flag,fisher_pval=FisherTest.FisherTest(pvals,alph)

    return fisher_pval/len(Z)

    #return flag







def getomega(data,X,Z):
    cov_m =np.cov(data,rowvar=False)
    col = list(data.columns)
    Xlist = []
    Zlist = []
    for i in X:
        t = col.index(i)
        Xlist.append(t)
    for i in Z:
        t = col.index(i)
        Zlist.append(t)
    B = cov_m[Xlist]
    B = B[:,Zlist]
    A = B.T
    u,s,v = np.linalg.svd(A)
    lens = len(X)
    omega =v.T[:,lens-1]
    omegalen=len(omega)
    omega=omega.reshape(1,omegalen)

    return omega


