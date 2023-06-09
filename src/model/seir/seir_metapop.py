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
        if beta is not None:
            self.beta = np.repeat([beta], n_t, axis=0)
        else:
            initial_betas = np.array([0.2, 2.5, 0.35, 3, 2, 1.7, 0.6, 3., 0.3, 2, 2.5,
                                      3., 3, 5.5, 2, 3.2, 5.1])
            end_betas = np.array([0.5, 0.5, 0.75, 9, 7, 3.7, 2, 3., 2., 5, 6.5,
                                      2., 3, 4.5, 2, 3.8, 5.1])
            beta = []
            for i in range(n_loc):
                beta.append(self.sigmoid(initial_betas[i], end_betas[i]))
            self.beta = np.transpose(np.array(beta))
        self.mu = mu
        self.Z = Z
        self.D = D
        self.alpha = alpha

        seed_loc = 2
        self.N = N
        self.E0 = E0 * np.ones((n_loc, n_loc))
        self.Ir0 = np.zeros((n_loc, n_loc))
        self.Iu0 = np.zeros((n_loc, n_loc))
        self.Iu0[seed_loc, seed_loc] = Iu0
        self.R0 = np.zeros((n_loc, n_loc))
        self.S0 = N - E0 - Iu0

        self.S, self.E, self.Ir, self.Iu, self.R, \
            self.i, self.i_true = self.gen_stoch_seir_metapop()

    def sigmoid(self, b_0, b_1):
        """Computes sigmoid curve"""
        t = np.arange(0, self.n_t)
        k = 0.1
        midpoint = 30
        sigmoid = b_0 + (b_1 - b_0) / (1 + np.exp(-k*(t - midpoint)))
        return sigmoid

    def gen_stoch_seir_metapop(
            self, add_noise=True, noise_param=1/50
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
                        t_index = np.minimum(np.floor(t).astype(int),
                                             self.n_t-1)
                        dSE = poisson(self.beta[t_index][i]*S[i, j] *
                                      (np.sum(Ir[:, i]) +
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
                        t_index = np.minimum(np.floor(t).astype(int),
                                             self.n_t-1)
                        dSE = poisson(self.beta[t_index][i]*S[i, j] *
                                      (np.sum(Ir[:, i]) +
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
            obs_error_var = np.maximum(1., i_true**2 * noise_param)
            for t in range(self.n_t):
                obs_error_sample = np.random.normal(0, 1, size=self.n_loc)
                i_true[t, :] += obs_error_sample * np.sqrt(obs_error_var[t, :])

            i_list = i_true.copy()

        return S_list, E_list, Ir_list, Iu_list, R_list, i_true, i_list

    def plot_state(self, ax=None, i=None):
        max_y = np.max(self.S) + 500
        ax.plot(self.S[:, i], '.-', label='S')
        ax.plot(self.E[:, i], '.-', label='E')
        ax.plot(self.Ir[:, i], '.-', label='Ir')
        ax.plot(self.Iu[:, i], '.-', label='Iu')
        ax.plot(self.R[:, i], '.-', label='R')
        ax.set_ylim(0, max_y)
        ax.set_xlabel("day")
        ax.set_ylabel("counts")
        ax.set_title(f'Location {i+1} SEIrIuR')
        ax.legend()

    def plot_obs(self, ax=None, i=None):
        max_y = np.max(self.i) + 700
        ax.plot(self.i[:, i], '-.')
        ax.set_ylim(0, max_y)
        ax.set_title(f'Location {i+1} Reported Daily Case Counts')
        ax.set_xlabel("day")
        ax.set_ylabel("daily case counts")

    def plot_all(self, path=None):
        fig, axs = plt.subplots(9, 4, sharex=True, figsize=(20, 30))
        for i, ax in enumerate(axs.reshape(-1)):
            if i > 33:
                ax.remove()
            elif i % 2 == 0:
                self.plot_state(ax, i//2)
            else:
                self.plot_obs(ax, i//2)

        if path:
            plt.savefig(f'{path}/synthetic_data.pdf')

    def save_data(self, path=None):
        # log source code
        lines = inspect.getsource(simualte_data)
        logging.info(lines)
        with open(f'{path}/data.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
