# This is the main program to measure Octupolar Susceptibility Chi_T at various temperature points
import numpy as np
from tqdm import tqdm
import newlib
import Coplanar_Configuration
import libChi_T^2
import matplotlib.pyplot as pl

# Set the temperature points where we want to measure the octupolar susceptibility
Temp_range = np.hstack([np.arange(0.0001,0.005,0.0003),np.arange(0.005,0.01,0.001),np.arange(0.01,0.02,0.003)])
# Flip the temperature range
Temp_range = np.flip(Temp_range)
ChiT = np.zeros_like(Temp_range)

# lattice size
L = 8
# Initialize a configuration by inputting J = 1 and the lattice size L
config = Coplanar_Configuration.Configuration(1.0, L)

n_cycles = 2000   # number of measurement cycles
n_warm = 10000    # number of hybrid Monte Carlo procedure for the "Warm up" process
n_sites = int(L**2*2)    # number of the lattice sites picked to flip the spin for the "Overrelaxation" procedure

# the cycle for all temperature points we picked
for i, T in enumerate(Temp_range):
    
    # the "Warm up" process to make the system reach equilibrium
    for n in range(n_warm):
        newlib.hybrid_Monte_Carlo(config,n_sites,1./T)
    # the measurement cycles
    for k in tqdm(range(n_cycles)):
        ChiT[i] += T*libChi_T^2.measureChiT(config,T,n_sites)/n_cycles
    
    # get physical quantities
    print(ChiT[i])
 
#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, ChiT)
