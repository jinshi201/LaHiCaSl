import numpy as np
import pandas as pd

def ToBij():
     ten = np.random.randint(0,2)
     s  =np.random.random()
     while abs(s) <0.5 and ten ==0:
        s  =np.random.random()
     result = ten+s
     if np.random.randint(0,10)>5:
        result = -1*result
     #print(result)
     return round(result,1)
     #return 1.2


def gen_coef():
    signed = 1
    if np.random.randn(1) >= 0:
        signed = -1
    return signed * np.random.uniform(.5, 2)


def SelectPdf(Num,data_type="exponential"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 5

    elif data_type == "gaussian":
        noise = np.random.normal(0, 1, size=Num)

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":  #exp-exponential
        noise = pow(np.random.exponential(scale=1, size=Num),2)

    elif data_type == "standard_exponential":
        noise =np.random.standard_exponential(size=Num)

    else: #uniform
        noise =np.random.uniform(-1, 1, size=Num)

    return noise




def SelectPdf2(Num,k=3):

    #noise = pow(np.random.uniform(low=-2.0,high=2.0,size=Num),3)
    #noise = pow(np.random.exponential(scale=1, size=Num),1)

    noise =np.random.standard_exponential(size=Num)
    #noise = np.random.exponential(scale=1, size=Num)
    #print('normal')
    #noise = np.random.normal(loc=0,scale=1,size=Num)

    return noise

def Toa():
    return 0.5


def CaseI(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L2

    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1
    x3=SelectPdf(Num)*Toa()+ToBij()*L1


    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2
    x6=SelectPdf(Num)*Toa()+ToBij()*L2



    x7=SelectPdf(Num)*Toa()+ToBij()*L3
    x8=SelectPdf(Num)*Toa()+ToBij()*L3
    x9=SelectPdf(Num)*Toa()+ToBij()*L3
    x10=SelectPdf(Num)*Toa()+ToBij()*L3+ToBij()*x9


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data


def CaseII(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)
    L3=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1
    L5=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1
    L6=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1
    L7=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L1


    x1=SelectPdf(Num)*Toa()+ToBij()*L3
    x2=SelectPdf(Num)*Toa()+ToBij()*L3


    x3=SelectPdf(Num)*Toa()+ToBij()*L4
    x4=SelectPdf(Num)*Toa()+ToBij()*L4



    x5=SelectPdf(Num)*Toa()+ToBij()*L5
    x6=SelectPdf(Num)*Toa()+ToBij()*L5


    x7=SelectPdf(Num)*Toa()+ToBij()*L6
    x8=SelectPdf(Num)*Toa()+ToBij()*L6

    x9=SelectPdf(Num)*Toa()+ToBij()*L7
    x10=SelectPdf(Num)*Toa()+ToBij()*L7


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data

def CaseIII(Num=3000):
    L1=SelectPdf(Num)

    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1


    L5=SelectPdf(Num)*Toa()+ToBij()*L2
    L6=SelectPdf(Num)*Toa()+ToBij()*L2

    L7=SelectPdf(Num)*Toa()+ToBij()*L3
    L8=SelectPdf(Num)*Toa()+ToBij()*L3

    L9=SelectPdf(Num)*Toa()+ToBij()*L4
    L10=SelectPdf(Num)*Toa()+ToBij()*L4

    x1=SelectPdf(Num)*Toa()+ToBij()*L5
    x2=SelectPdf(Num)*Toa()+ToBij()*L5

    x3=SelectPdf(Num)*Toa()+ToBij()*L6
    x4=SelectPdf(Num)*Toa()+ToBij()*L6

    x5=SelectPdf(Num)*Toa()+ToBij()*L7
    x6=SelectPdf(Num)*Toa()+ToBij()*L7

    x7=SelectPdf(Num)*Toa()+ToBij()*L8
    x8=SelectPdf(Num)*Toa()+ToBij()*L8

    x9=SelectPdf(Num)*Toa()+ToBij()*L9
    x10=SelectPdf(Num)*Toa()+ToBij()*L9

    x11=SelectPdf(Num)*Toa()+ToBij()*L10
    x12=SelectPdf(Num)*Toa()+ToBij()*L10

    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data

def CaseIV(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)*Toa()+ToBij()*L1
    L3=SelectPdf(Num)*Toa()+ToBij()*L1
    L4=SelectPdf(Num)*Toa()+ToBij()*L1


    L5=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L3
    L6=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*L3
    L7=SelectPdf(Num)*Toa()+ToBij()*L3
    L8=SelectPdf(Num)*Toa()+ToBij()*L1+ToBij()*L4
    L9=SelectPdf(Num)*Toa()+ToBij()*L4


    x1=SelectPdf(Num)*Toa()+ToBij()*L5+ToBij()*L6
    x2=SelectPdf(Num)*Toa()+ToBij()*L5+ToBij()*L6
    x3=SelectPdf(Num)*Toa()+ToBij()*L5+ToBij()*L6

    x4=SelectPdf(Num)*Toa()+ToBij()*L7
    x5=SelectPdf(Num)*Toa()+ToBij()*L7

    x6=SelectPdf(Num)*Toa()+ToBij()*L8
    x7=SelectPdf(Num)*Toa()+ToBij()*L8

    x8=SelectPdf(Num)*Toa()+ToBij()*L9
    x9=SelectPdf(Num)*Toa()+ToBij()*L9
    x10=SelectPdf(Num)*Toa()+ToBij()*L9

    x11=SelectPdf(Num)*Toa()+ToBij()*L2
    x12=SelectPdf(Num)*Toa()+ToBij()*L4

    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data



def ImpureGdata(Num=3000):
    L1=SelectPdf(Num)
    L2=SelectPdf(Num)+ToBij()*L1
    L3=SelectPdf(Num)+ToBij()*L1

    x1=SelectPdf(Num)*Toa()+ToBij()*L1
    x2=SelectPdf(Num)*Toa()+ToBij()*L1
    x3=SelectPdf(Num)*Toa()+ToBij()*L1


    x4=SelectPdf(Num)*Toa()+ToBij()*L2
    x5=SelectPdf(Num)*Toa()+ToBij()*L2
    x6=SelectPdf(Num)*Toa()+ToBij()*L2
    x10=SelectPdf(Num)*Toa()+ToBij()*L2+ToBij()*x6



    x7=SelectPdf(Num)*Toa()+ToBij()*L3+ToBij()*L2
    x8=SelectPdf(Num)*Toa()+ToBij()*L3+ToBij()*L2
    x9=SelectPdf(Num)*Toa()+ToBij()*L3+ToBij()*L2

    x11=SelectPdf(Num)*Toa()+ToBij()*L3
    x12=SelectPdf(Num)*Toa()+ToBij()*L3


    data = pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]).T,columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])

    data = (data-data.mean())/data.std()
    #data = data-data.mean()
    return data



def main():
    data = CaseI(1000)
    print(data)
    #data.to_csv('data.csv',header=0,index=0)

if __name__ == '__main__':
    main()
