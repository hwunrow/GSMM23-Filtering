import math
import random
import numpy as np

true_params = {
            'beta': 2.1,
            'mu': 0.5,
            'Z': 4,
            'D': 4.,
            'alpha': 0.3,
            'n_t': 100,
            'N': 100_000.,
            'E0': 0.,
            'Iu0': 10.,
        }


def forecast(t, x, N, dt=1, noise_param=1/25):
    """
    forecast step
    """
    S = x[0]
    E = x[1]
    Ir = x[2]
    Iu = x[3]
    R = x[4]
    # Stochastic transitions
    dSE = np.random.poisson(true_params["beta"]*S*(Ir+true_params['mu']*Iu)/true_params['N'])
    dEI = np.random.poisson(E/true_params['Z'])
    dIrR = np.random.poisson(Ir/true_params['D'])
    dIuR = np.random.poisson(Iu/true_params['D'])

    S_new = np.clip(S-dSE, 0, true_params['N'])
    E_new = np.clip(E+dSE-dEI, 0, true_params['N'])
    Ir_new = np.clip(Ir+dEI*true_params['alpha']-dIrR, 0, true_params['N'])
    Iu_new = np.clip(Iu+dEI*(1-true_params['alpha'])-dIuR, 0, true_params['N'])
    R_new = np.clip(R+dIrR+dIuR, 0, true_params['N'])
    
    # x_new = np.concatenate((S_new, E_new, Ir_new, Iu_new, R_new, dEI*true_params['alpha']))
    x_new = np.row_stack((S_new, E_new, Ir_new, Iu_new, R_new, dEI*true_params['alpha']))
    return x_new


def f0(N, m=300):
    """
    Initial guess of the state space.
        Args:
            N: population
            m: number of ensemble members
    """
    S0 = np.random.uniform(N*0.8, N, size=m)
    E0 = np.zeros(m)
    Ir0 = np.zeros(m)
    Iu0 = N - S0
    R0 = np.zeros(m)
    i0 = np.zeros(m)
 
    # x = np.concatenate((S0, E0, Ir0, Iu0, R0, i0))
    x=np.row_stack((S0, E0, Ir0, Iu0, R0, i0))
    return x

def resample(w, I):
    w=[0]*6
    w_hat=[0]*6
    N=len(w)
    for j in range(1,N):
        w_hat[j]=sum(w[0:j+1])
    u=random.uniform(0, 1/N)
    k=0
    for j in range(N):
        while u>w_hat[k]:
            k=k+1
            if k==len(w_hat):
                break
        I[j]=k
        u=u+1/N
        k=1
# C: random measurement noise
# H(z): model function provided by modeler
C=np.maximum(1,3**2*(1/50))
def H(x):
    """
    Observational function.
        Args:
            x: state space
    """
    return x[-1]

def likelihood(d,z_j):
    
    if np.shape(C):

        detC=np.linalg.det(C)
        invC=np.linalg.inv(C)
        a=np.dot((d-H(z_j)),invC)
        a=np.dot(a,(d-H(z_j)))
    else:
        detC=C
        invC=1/C
        a=(d-H(z_j))*invC
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
        try:
            del_w[j]=math.log(likelihood(d,z[:,j])) 
        except:
            # print(likelihood(d,z[:,j]))
            del_w[j]=0
    del_w_max=max(del_w)
    for j in range(N):
        del_w[j]=math.exp(del_w[j]-del_w_max)
    sum_w=sum(del_w)
    for j in range(N):
        w[j]=del_w[j]/sum_w
    I=[0]*N
    resample(w,I)
    for j in range(N):
        z[:,j]=z[:,I[j]]


if __name__ == "__main__":
    

    np.random.seed(1994)
    data = seir.simualte_data(**true_params, add_noise=True, noise_param=1/50)
    data.plot_all(path='./figures')

    with open('../model/seir/test_data.npy', 'wb') as f:
        np.save(f, data.i)
    x=f0(100)
    N=100000
    pt=[]
    # import pdb; pdb.set_trace()
    x1=forecase(t,x,N)
    for t in range(1, 100):
        # import pdb; pdb.set_trace()
        x1=forecase(t,x1,N)
        # print('before')
        # print(x1)
        try:
            basic_p_filter(x1,data.i[t])
            pt.append(x1[0][2])
            # print('after')
            # print(x1)
        except:
            print(t)
   
 
    ts=np.linspace(0,101,num=100)
    plt.clf()
    plt.plot(ts,pt)
    plt.show()
