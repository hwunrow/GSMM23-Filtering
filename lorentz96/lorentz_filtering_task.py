from lorentz_96 import L96_full
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


num_dim = 100
num_t_steps = 10001
t_vals = np.linspace(0, 1, num_t_steps)
true_IC = 3.0*(np.random.rand(num_dim,) - 0.5)


def data_rhs(x, t):
    return L96_full(x, t, num_dim)


full_model = [true_IC]

for step_index in range(0, num_t_steps - 1):
    temp_times = [t_vals[step_index], t_vals[step_index + 1]]
    ode_results = odeint(data_rhs, full_model[step_index], temp_times)
    new_step = ode_results[1]
    noise = np.random.normal(0, 1/10.0*np.abs(new_step))
    final_result = new_step + noise
    full_model.append(final_result)

# plot random component to get an idea what the noise looks like, after saving to csv
df = pd.DataFrame(full_model).set_index(t_vals)
df.to_csv("lor_96.csv")

plt.plot(t_vals, df.loc[:,3])
plt.show()
