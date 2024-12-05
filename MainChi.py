import numpy as np
from tqdm import tqdm
import newlib
import classConfiguration2
import libChiT2
import matplotlib.pyplot as pl



# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.005,0.0003),np.arange(0.005,0.01,0.001),np.arange(0.01,0.02,0.003)])
# Temp_range = np.flip(Temp_range)#
ChiT = np.zeros_like(Temp_range)

# lattice size
# lattice size
L = 8
config = classConfiguration2.Configuration(1.0, L)
#pbar = tqdm_notebook(Temp_range)
n_cycles = 2000
n_cool = 10000
n_warm = 10000
n_sites = int(L**2*2)
# for n in range(n_cool):
#     newlib.hybrid_Monte_Carlo(config,n_sites,1./0.01)
for i, T in enumerate(Temp_range):
    # Process the Hybrid MC, and get the average value
    for n in range(n_warm):
        newlib.hybrid_Monte_Carlo(config,n_sites,1./T)
    for k in tqdm(range(n_cycles)):
        ChiT[i] += T*libChiT2.measureChiT(config,T,n_sites)/n_cycles

    # get physical quantities
    #print(T*ChiT[i])
 
#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, T*ChiT)

# pl.semilogy(Temp_range,ChiT)
# pl.show()