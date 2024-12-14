# This is the main program to measure the octuple tensor (T_alpha_beta_gamma)^2 at various temperature points
import numpy as np
from tqdm import tqdm
import newlib
import classConfiguration
import libChi_T^2
import matplotlib.pyplot as pl

# Set the temperature points where we want to measure the octuple tensor
Temp_range = np.hstack([np.arange(0.0001,0.001,0.00003),np.arange(0.001,0.01,0.0003)])
# Flip the temperature range
Temp_range = np.flip(Temp_range)
T2 = np.zeros_like(Temp_range)

# lattice size
L = 12
# Initialize a configuration by inputting J = 1 and the lattice size L
config = classConfiguration.Configuration(1.0, L)

n_cycles = 5000    # number of measurement cycles
n_cool = 6000    # number of hybrid Monte Carlo procedure for the "Cool" process to achieve the initial equilibrium
n_warm = 6000    # number of hybrid Monte Carlo procedure for the "Warm up" process
n_sites = int(L**2*2)    # number of the lattice sites picked to flip the spin for the "Overrelaxation" procedure

# the "Cool" process to achieve the initial equilibrium
for n in range(n_cool):
    newlib.hybrid_Monte_Carlo(config,n_sites,1./0.01)

# the cycle for all temperature points we picked
for i, T in enumerate(Temp_range):
    
    # the "Warm up" process to make the system reach equilibrium at every temperature
    for n in range(n_warm):
        newlib.hybrid_Monte_Carlo(config,n_sites,1./T)
    # the measurement cycles
    for k in tqdm(range(n_cycles)):
        T2[i] += libChi_T^2.T2S(config,n_sites,1./T)/n_cycles

    # get physical quantities
    print(T,T2[i])

#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, T2)
