import math
import random
import numpy as np

def resample(w,I):
    w=[0]*6
    w_hat=[0]*6
    N=len(w)
    for j in range(1,N):
        w_hat[j]=sum(w[0:j+1])
    u=random.rand(0,1/N)
    k=0
    for j in range(N):
        while u>w_hat[k]:
            k=k+1
        I[j]=k
        u=u+1/N
        k=1
# C: random measurement noise
# H(z): model function provided by modeler
C=[1,1,1]
def H(z):
    pass

def likelihood(d,z_j):
    detC=np.linalg.det(C)
    invC=np.linalg.inv(C)
    a=np.dot((d-H(z_j)),invC)
    a=np.dot(a,(d-H(z_j)))
    b=2*math.pi*detC
    a=(-1/2)*a/math.sqrt(b)
    return math.exp(a)

def basic_p_filter(z,d):
    # z is the list of all points, nxN
    # d is the measurement vector, m
    N=len(z[0])
    w=[0]*N
    del_w=[0]*N
    for j in range(N):
        del_w[j]=math.log(likelihood(d,z[j])) 
    del_w_max=max(del_w)
    for j in range(N):
        del_w[j]=math.exp(del_w[j]-del_w_max)
    sum_w=sum(del_w)
    for j in range(N):
        w[j]=del_w[j]/sum_w
    I=[0]*N
    resample(w,I)
    for j in range(N):
        z[j]=z[I[j]]


