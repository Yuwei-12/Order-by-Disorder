import numpy as np
from tqdm import tqdm
import lib_capacity
import classConfiguration
import libChiT2
import matplotlib.pyplot as pl



# initialize a configuration
Temp_range = np.hstack([0.0005,np.arange(0.001,0.01,0.001),np.arange(0.01,0.1,0.015)])
T2 = np.zeros_like(Temp_range)
print(Temp_range)

# lattice size
L = 12

#pbar = tqdm_notebook(Temp_range)
for i, T in enumerate(Temp_range):
    #pbar.set_description(f"Processing T = {T:.2f}")
    # at each temperature, initialize the Configuration
    config = classConfiguration.Configuration(T, 1.0, L)
    n_cycles = 50000
    n_warmup = 10000
    n_sites = int(L**2/3)
    # Process the Hybrid MC, and get the average value
    for n in range(n_warmup):
        num_of_over_relax = 4
        lib_capacity.hybrid_Monte_Carlo(config,n_sites)
    #Ttensor = np.zeros((3, 3, 3))
    S3 = 0
    S = 0
    for k in tqdm(range(n_cycles)):
        S3 += libChiT2.T2S3_measure(config,n_sites)
        S += libChiT2.T2S_measure(config,n_sites)
    S3 /= float(n_cycles)
    S /= float(n_cycles)
    # get physical quantities
    T2[i] = (S3 - 0.6*S)/((3 * L**2)**2)

#Save quantities in a file
np.savetxt("Tabc_%i.dat"%L, T2)

pl.semilogx(Temp_range,T2)
pl.show()
