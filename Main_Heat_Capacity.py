# This is the main program to measure heat capacity at various temperature points
import numpy as np
from tqdm import tqdm
import newlib
import classConfiguration

# Set the temperature points where we want to measure the heat capacity
Temp_range = np.hstack([np.arange(0.0001,0.01,0.00033),np.arange(0.01,0.1,0.015),np.arange(0.1,0.4,0.1)])
# Flip the temperature range
Temp_range = np.flip(Temp_range)
Capacity = np.zeros_like(Temp_range)

# lattice size
L = 8
# Initialize a configuration by inputting J = 1 and the lattice size L
config = classConfiguration.Configuration(1.0, L)
config_2 = config

# the cycle for all temperature points we picked
for i, T in tqdm(enumerate(Temp_range)):
    # Transfer the configuration
    config_1 = config_2
    av_e, av_e2 = 0, 0
    n_cycles = 10000   # number of measurement cycles
    n_warmup = 6000   # number of hybrid Monte Carlo procedure for the "Warm up" process
    n_sites = int(L**2*2/3)   # number of the lattice sites picked to flip the spin for the "Overrelaxation" procedure
    
    # the "Warm up" process to make the system reach equilibrium
    for n in range(n_warmup):
        newlib.hybrid_Monte_Carlo(config_1, T, n_sites)
    
    # the measurement cycles
    for k in range(n_cycles):
        e0 = newlib.measure(config_1, n_sites, T)
        av_e += e0   # the sum of energy E
        av_e2 += e0 ** 2   # the sum of E^2
    
    # transfer the configuration to the next temperature
    config_2 = config_1
    # calculate the averages
    av_e /= float(n_cycles)
    av_e2 /= float(n_cycles)

    # get physical quantities
    Capacity[i] = (av_e2 - av_e**2)/((T**2)*(L**2)*3)
    print(av_e**2,av_e2,Capacity[i])

#Save quantities in a file
np.savetxt("Capacity_%i.dat"%L, Capacity)
