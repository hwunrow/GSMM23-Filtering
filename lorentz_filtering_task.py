from lorentz_96 import L96_full
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


num_dim = 100
t_vals = np.linspace(0, 1, 1001)
true_IC = 3.0*(np.random.rand(num_dim,) - 0.5)


def data_rhs(x, t):
    return L96_full(x, t, num_dim)


full_model = odeint(data_rhs, true_IC, t_vals)
noisy_model = full_model + np.random.normal(0, 1/10.0*np.abs(full_model), (1001, num_dim))

# plot a randomly selected component of full model, noisy model
plt.plot(t_vals, noisy_model[:, 33])
plt.plot(t_vals, full_model[:, 33])
plt.show()

df = pd.DataFrame(full_model).set_index(t_vals)
df.to_csv("lor_96.csv")
