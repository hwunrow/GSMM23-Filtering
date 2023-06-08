from scipy.integrate import odeint
import numpy as np


# basic code for right_hand side found here:
# https://en.wikipedia.org/wiki/Lorenz_96_model


def L96_full(x, t, N):
    epsilon = 0.1
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - epsilon*x[i - 2]) * x[i - 1] - x[i] + 0.1
    return d


def L96_broken(x, t, N):
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = x[(i + 1) % N] * x[i - 1] - x[i] + 0.1
    return d


def step_particles(x0_particles, dt, problem_dimension=100):
    t_vals = [0.0, dt]  # a hacky approach to take one step with a numerical integrator
    particle_positions = []

    def model_rhs(x, t):
        return L96_broken(x, t, problem_dimension)

    for init_cond in x0_particles:
        particle_step = odeint(model_rhs, init_cond, t_vals)
        particle_positions.append(particle_step[1])
    return particle_positions
