import numpy as np
from tqdm import tqdm
import lib
import classConfiguration
import libChiT2
import matplotlib.pyplot as pl



# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.01,0.00033),np.arange(0.0105,0.1,0.015),np.arange(0.13,0.4,0.1)])
ChiT = np.zeros_like(Temp_range)

# lattice size
L = 12

#pbar = tqdm_notebook(Temp_range)
for i, T in enumerate(Temp_range):
    #pbar.set_description(f"Processing T = {T:.2f}")
    # at each temperature, initialize the Configuration
    config = classConfiguration.Configuration(T, 1.0, L)
    n_cycles = 5*100
    n_warmup = 10000
    n_sites = int(L**2/3)
    # Process the Hybrid MC, and get the average value
    for n in tqdm(range(n_warmup)):
        num_of_over_relax = 4
        lib.hybrid_Monte_Carlo(config,n_sites)
    for k in tqdm(range(n_cycles)):
        ChiT[i] += libChiT2.measureChiT(config,T,n_sites)/n_cycles

    # get physical quantities
    print(T*ChiT[i])
 
#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, ChiT)

pl.semilogx(Temp_range,ChiT)
pl.show()