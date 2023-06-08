import numpy as np
import matplotlib.pyplot as plt
import pickle
import inspect
import logging


class simualte_data():
    def __init__(
            self, n_t, beta, mu, Z, D, alpha, N, E0, Iu0, **kwargs
            ):
        r"""
        Args:
          n_t (int): number of days
          beta (float): transmission rate
          mu (float): relative transmission rate of unreported
          Z (float): avg latency period
          D (float): avg infectious period
          alpha (float): reporting proportion
          N (int): Population
          E0 (int): Iniital number of exposed
          Iu0 (int): Intitial number of unreported infectors
        """
        self.n_t = n_t

        self.beta = beta
        self.mu = mu
        self.Z = Z
        self.D = D
        self.alpha = alpha

        self.N = N
        self.S0 = (N - E0 - Iu0)
        self.E0 = E0 
        self.Ir0 = 0
        self.Iu0 = Iu0
        self.R0 = 0

        self.S, self.E, self.Ir, self.Iu, self.R, \
            self.i, self.i_true = self.gen_stoch_seir()

    def gen_stoch_seir(
            self, add_noise=True, noise_param=1/50
            ):
        S = np.array([self.S0])
        E = np.array([self.E0])
        Ir = np.array([self.Ir0])
        Iu = np.array([self.Iu0])
        R = np.array([self.R0])
        i = np.array([0])
        for t in range(self.n_t):
            dSE = np.random.poisson(self.beta*S[t]*(Ir[t]+self.mu*Iu[t])/self.N)
            dEI = np.random.poisson(E[t]/self.Z)
            dIrR = np.random.poisson(Ir[t]/self.D)
            dIuR = np.random.poisson(Iu[t]/self.D)

            S_new = np.clip(S[t]-dSE, 0, self.N)
            E_new = np.clip(E[t]+dSE-dEI, 0, self.N)
            Ir_new = np.clip(Ir[t]+dEI*self.alpha-dIrR, 0, self.N)
            Iu_new = np.clip(Iu[t]+dEI*(1-self.alpha)-dIuR, 0, self.N)
            R_new = np.clip(R[t]+dIrR+dIuR, 0, self.N)

            S = np.append(S, S_new)
            E = np.append(E, E_new)
            Ir = np.append(Ir, Ir_new)
            Iu = np.append(Iu, Iu_new)
            R = np.append(R, R_new)
            i = np.append(i, dEI*self.alpha)

        i_true = i
        if add_noise:
            i = i.astype('float64')
            self.noise_param = noise_param
            obs_error_var = np.maximum(1., i[1:]**2 * noise_param)
            obs_error_sample = np.random.normal(0, 1, size=self.n_t)
            i[1:] += obs_error_sample * np.sqrt(obs_error_var)
            i = np.clip(i, 0, self.N)

        return S, E, Ir, Iu, R, i, i_true

    def plot_state(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.S, '.-', label='S')
        ax.plot(self.E, '.-', label='E')
        ax.plot(self.Ir, '.-', label='Ir')
        ax.plot(self.Iu, '.-', label='Iu')
        ax.plot(self.R, '.-', label='R')
        ax.set_title('Stochastic SEIrIuR')
        ax.legend()

    def plot_obs(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.i, '.')
        ax.set_title('Stochastic Daily Case Counts')

    def plot_all(self, path=None):
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        self.plot_state(axs[0])
        self.plot_obs(axs[1])

        if path:
            plt.savefig(f'{path}/synthetic_data.pdf')

    def save_data(self, path=None):
        # log source code
        lines = inspect.getsource(simualte_data)
        logging.info(lines)
        with open(f'{path}/data.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
