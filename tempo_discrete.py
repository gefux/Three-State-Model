import sys
import numpy as np
sys.path.insert(0,'..')

import time_evolving_mpo as tempo
import numpy as np
import matplotlib.pyplot as plt
from bath_para import c_list, w_list


tempo.__version__

# System Hamiltonian
coupling_mat = np.array([[0, 300, 0],
                        [300, 0, 50],
                        [0, 50 , 0]])
h1e = coupling_mat + np.diag([0,1613.1385305,219.47463]) + np.diag([1941.303314406233, 327.4029963751762, 599.7535213179169])

system = tempo.System(h1e)

# Build a correlation function from the discrete couplings and frequencies
corr_single = lambda c, w, t, T:  c**2 * (np.cos(w*t) / np.tanh(w/2/T) - 1j * np.sin(w*t))
corr = lambda t, T: np.sum([corr_single(c, w, t, T) for c, w in zip(c_list, w_list)])

# Temperature 300K, in a particular system of units
temperature = 0.69*300
custom_corr = lambda t: corr(t, temperature)

correlations = tempo.CustomCorrelations(custom_corr)

# System-bath coupling: the system operator
sys_coup = np.diag([1, 0.4106715589673543, -0.5558270843261847])

bath = tempo.Bath(sys_coup, correlations)

tempo_parameters = tempo.TempoParameters(dt=0.0001, dkmax=4000, epsrel=10**(-4))

# 1-picosencond-long dynamics, in a particular unit system.
dynamics = tempo.tempo_compute(system=system,
                               bath=bath,
                               initial_state=np.diag([1,0,0]),
                               start_time=0.0,
                               end_time=0.19,
                               parameters=tempo_parameters)

# Measure the population on state 1
population1 = np.diag([1,0,0])
t, p1 = dynamics.expectations(population1, real=True)

print(t)
print(list(p1))

plt.plot(t, p1, label=r'$P_1$')
plt.xlabel(r'$t$')
plt.ylabel('Population 1')
plt.legend()
plt.show()
