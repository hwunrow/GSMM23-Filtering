import numpy as np
from numpy.random import poisson
import matplotlib.pyplot as plt
import pickle
import inspect
import logging


class simualte_data():
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
            self, add_noise=False, noise_param=1/50
            ):
        S = self.S0
        E = self.E0
        Ir = self.Ir0
        Iu = self.Iu0
        R = self.R0

        S_list = np.sum(self.S0, axis=1)
        E_list = np.sum(self.E0, axis=1)
        Ir_list = np.sum(self.Ir0, axis=1)
        Iu_list = np.sum(self.Iu0, axis=1)
        R_list = np.sum(self.R0, axis=1)
        i_true = np.zeros(self.n_loc)
        i_list = np.zeros(self.n_loc)

        # Time
        dt_day = 1 / 3

        # Generate tspan
        tspan = np.zeros(2 * self.n_t+1)
        for i in range(self.n_t):
            tspan[2*i] = i
            tspan[2*i+1] = i + dt_day
        tspan[-1] = self.n_t

        for t in tspan:
            if t % 1 == 0:  # daytime
                daytime_i = np.zeros(self.n_loc)
                daytime_i = np.zeros(self.n_loc)
                # Loop over destination
                for i in range(self.n_loc):
                    # Day time population
                    N_i = np.sum(self.N[i, :]) + np.sum(Ir[:, i]) - \
                        np.sum(Ir[i, :])

                    # Loop over origin
                    for j in range(self.n_loc):
                        dSE = poisson(self.beta[i]*S[i, j]*(np.sum(Ir[:, i]) +
                                      self.mu*np.sum(Iu[i, :]))/N_i)
                        dEI = poisson(E[i, j]/self.Z)
                        dIrR = poisson(Ir[i, j]/self.D)
                        dIuR = poisson(Iu[i, j]/self.D)

                        S[i, j] = np.clip(S[i, j] - dSE,
                                          0, np.sum(self.N[i, :]))
                        E[i, j] = np.clip(E[i, j] + dSE - dEI,
                                          0, np.sum(self.N[i, :]))
                        Ir[i, j] = np.clip(Ir[i, j] + dEI*self.alpha - dIrR,
                                           0, np.sum(self.N[i, :]))
                        Iu[i, j] = np.clip(Iu[i, j] + dEI*(1-self.alpha) -
                                           dIuR,
                                           0, np.sum(self.N[i, :]))
                        R[i, j] = np.clip(R[i, j] + dIrR + dIuR,
                                          0, np.sum(self.N[i, :]))

                        daytime_i[i] += dEI*self.alpha

            else:  # nighttime
                nighttime_i = np.zeros(self.n_loc)
                # loop over destination
                for i in range(self.n_loc):
                    # loop over origin
                    for j in range(self.n_loc):
                        # night time population
                        N_j = np.sum(self.N[:, j])
                        dSE = poisson(self.beta[i]*S[i, j]*(np.sum(Ir[:, i]) +
                                      self.mu*np.sum(Iu[i, :]))/N_j)
                        dEI = poisson(E[i, j]/self.Z)
                        dIrR = poisson(Ir[i, j]/self.D)
                        dIuR = poisson(Iu[i, j]/self.D)

                        S[i, j] = np.clip(S[i, j] - dSE,
                                          0, np.sum(self.N[i, :]))
                        E[i, j] = np.clip(E[i, j] + dSE - dEI,
                                          0, np.sum(self.N[i, :]))
                        Ir[i, j] = np.clip(Ir[i, j] + dEI*self.alpha - dIrR,
                                           0, np.sum(self.N[i, :]))
                        Iu[i, j] = np.clip(Iu[i, j] + dEI*(1-self.alpha) -
                                           dIuR, 0, np.sum(self.N[i, :]))
                        R[i, j] = np.clip(R[i, j] + dIrR + dIuR,
                                          0, np.sum(self.N[i, :]))

                        nighttime_i[i] += dEI*self.alpha
                S_list = np.vstack([S_list, np.sum(S, axis=1)])

                E_list = np.vstack([E_list, np.sum(E, axis=1)])
                Ir_list = np.vstack([Ir_list, np.sum(Ir, axis=1)])
                Iu_list = np.vstack([Iu_list, np.sum(Iu, axis=1)])
                R_list = np.vstack([R_list, np.sum(R, axis=1)])
                i_true = np.vstack([i_true, daytime_i + nighttime_i])

            if add_noise:
                self.noise_param = noise_param
                obs_error_var = np.maximum(1.,
                                           np.array(i_true)**2 * noise_param)
                obs_error_sample = np.random.normal(0, 1, size=self.n_t)
                i_true += obs_error_sample * np.sqrt(obs_error_var)
                i_list = np.clip(i_true, 0, self.N)

        return S_list, E_list, Ir_list, Iu_list, R_list, i_true, i_list

    def plot_state(self, axs=None):
        for i, ax in enumerate(axs):
            ax.plot(self.S[:, i], '.-', label='S')
            ax.plot(self.E[:, i], '.-', label='E')
            ax.plot(self.Ir[:, i], '.-', label='Ir')
            ax.plot(self.Iu[:, i], '.-', label='Iu')
            ax.plot(self.R[:, i], '.-', label='R')
            ax.set_title(f'Location {i} Stochastic SEIrIuR')
            ax.legend()

    def plot_obs(self, axs=None):
        for i, ax in enumerate(axs):
            ax.plot(self.i[:, i], '.')
            ax.set_title(f'Location {i} Stochastic Daily Case Counts')

    def plot_all(self, path=None):
        fig, axs = plt.subplots(self.n_loc, 2, sharex=True, figsize=(10, 50))
        self.plot_state(axs[:, 0])
        self.plot_obs(axs[:, 1])

        if path:
            plt.savefig(f'{path}/synthetic_data.pdf')

    def save_data(self, path=None):
        # log source code
        lines = inspect.getsource(simualte_data)
        logging.info(lines)
        with open(f'{path}/data.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
