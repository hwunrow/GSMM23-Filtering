import numpy as np
import matplotlib.pyplot as plt
import pickle
import inspect
import logging


class seir_metapop():
    def __init__(
            self, n_t, n_loc, beta, mu, Z, D, alpha, N, E0, Iu0, **kwargs
            ):
        r"""
        Args:
          n_t (int): number of days
          n_loc (int): number of locations
          beta (list): transmission rates
          mu (float): relative transmission rate of unreported
          Z (float): avg latency period
          D (float): avg infectious period
          alpha (float): reporting proportion
          N (int): Population
          E0 (int): Iniital number of exposed
          Iu0 (int): Intitial number of unreported infectors
        """
        self.n_t = n_t
        self.n_loc = n_loc

        self.beta = beta
        if len(beta) != n_loc:
            raise Exception(f"there should be {n_loc} beta")
        self.mu = mu
        self.Z = Z
        self.D = D
        self.alpha = alpha

        self.N = N
        self.S0 = (N - E0 - Iu0) * np.ones((n_loc, n_loc))
        self.E0 = E0 * np.ones((n_loc, n_loc))
        self.Ir0 = np.zeros((n_loc, n_loc))
        self.Iu0 = Iu0 * np.ones((n_loc, n_loc))
        self.R0 = np.zeros((n_loc, n_loc))

        self.S, self.E, self.Ir, self.Iu, self.R, \
            self.i, self.i_true = self.gen_stoch_seir_metapop()

    def gen_stoch_seir_metapop(
            self, add_noise=False, noise_param=1/25
            ):
        S = self.S0
        E = self.E0
        Ir = self.Ir0
        Iu = self.Iu0
        R = self.R0

        S_list = [S]
        E_list = [E]
        Ir_list = [Ir]
        Iu_list = [Iu]
        R_list = [R]
        i_list = np.zeros([0])

        # Time
        dt_day = 1 / 3

        # Generate tspan
        tspan = np.zeros(2 * self.n_t+1)
        for i in range(self.n_t):
            tspan[2*i] = i
            tspan[2*i+1] = i + dt_day
        tspan[-1] = self.n_t

        daytime_i = 0
        for t in tspan:
            if t % 1 == 0:  # daytime
                # Loop over destination
                for i in range(self.n_loc):
                    # Day time population
                    N_i = np.sum(self.N[i, :]) + np.sum(Ir[:, i]) - np.sum(Ir[i, :])

                    # Loop over origin
                    for j in range(self.n_loc):
                        dSE = np.random.poisson(self.beta[i]*S[i, j]*(np.sum(Ir[:, i])+self.mu*np.sum(Iu[i, :]))/N_i)
                        dEI = np.random.poisson(E[i, j]/self.Z)
                        dIrR = np.random.poisson(Ir[i, j]/self.D)
                        dIuR = np.random.poisson(Iu[i, j]/self.D)

                        S = np.clip(S-dSE, 0, self.N)
                        E = np.clip(E+dSE-dEI, 0, self.N)
                        Ir = np.clip(Ir+dEI*self.alpha-dIrR, 0, self.N)
                        Iu = np.clip(Iu+dEI*(1-self.alpha)-dIuR, 0, self.N)
                        R = np.clip(R+dIrR+dIuR, 0, self.N)

                        S_list = np.append(S_list, S)
                        E_list = np.append(E_list, E)
                        Ir_list = np.append(Ir_list, Ir)
                        Iu_list = np.append(Iu_list, Iu)
                        R_list = np.append(R_list, R)
                        daytime_t+= dEI*self.alpha

            else:  # nighttime
                # loop over destination
                for i in range(self.n_loc):
                    # loop over origin
                    for j in range(self.n_loc):
                        # night time population
                        N_j = np.sum(self.N[:, j])
                        dSE = np.random.poisson(self.beta[i]*S[i, j]*(np.sum(Ir[:, i])+self.mu*np.sum(Iu[i, :]))/N_j)
                        dEI = np.random.poisson(E[i, j]/self.Z)
                        dIrR = np.random.poisson(Ir[i, j]/self.D)
                        dIuR = np.random.poisson(Iu[i, j]/self.D)

                        S = np.clip(S-dSE, 0, self.N)
                        E = np.clip(E+dSE-dEI, 0, self.N)
                        Ir = np.clip(Ir+dEI*self.alpha-dIrR, 0, self.N)
                        Iu = np.clip(Iu+dEI*(1-self.alpha)-dIuR, 0, self.N)
                        R = np.clip(R+dIrR+dIuR, 0, self.N)

                        S_list = np.append(S_list, S)
                        E_list = np.append(E_list, E)
                        Ir_list = np.append(Ir_list, Ir)
                        Iu_list = np.append(Iu_list, Iu)
                        R_list = np.append(R_list, R)
                        
                    i_list = np.append(i_list, dEI*self.alpha)

                    if add_noise:
                        i = i.astype('float64')
                        self.noise_param = noise_param
                        obs_error_var = np.maximum(1., i[1:]**2 * noise_param)
                        obs_error_sample = np.random.normal(0, 1, size=self.n_t)
                        i[1:] += obs_error_sample * np.sqrt(obs_error_var)
                        i = np.clip(i, 0, self.N)

        return S_list, E_list, Ir_list, Iu_list, R_list, i_list, i_true

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
        lines = inspect.getsource(seir_metapop)
        logging.info(lines)
        with open(f'{path}/data.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)