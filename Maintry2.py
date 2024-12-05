import numpy as np
from tqdm import tqdm
import newlib
import classConfiguration
import libChiT2
import matplotlib.pyplot as pl



# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.001,0.00003),np.arange(0.001,0.01,0.0003)])
Temp_range = np.flip(Temp_range)
T2 = np.zeros_like(Temp_range)

# lattice size
# lattice size
L = 12
config = classConfiguration.Configuration(1.0, L)
#pbar = tqdm_notebook(Temp_range)
n_cycles = 5000
n_cool = 6000
n_warm = 6000
n_sites = int(L**2*2)
for n in range(n_cool):
    newlib.hybrid_Monte_Carlo(config,n_sites,1./0.01)
for i, T in enumerate(Temp_range):
    # Process the Hybrid MC, and get the average value
    for n in range(n_warm):
        newlib.hybrid_Monte_Carlo(config,n_sites,1./T)
    for k in tqdm(range(n_cycles)):
        T2[i] += libChiT2.T2S(config,n_sites,1./T)/n_cycles

    # get physical quantities
    print(T,T2[i])
#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, T2)

pl.plot(Temp_range,T2)
pl.show()