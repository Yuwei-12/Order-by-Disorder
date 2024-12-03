import numpy as np
from tqdm import tqdm
import lib
import classConfiguration
import matplotlib.pyplot as pl


# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.01,0.00015),np.arange(0.01,0.1,0.015),np.arange(0.1,0.4,0.1)])
Temp_range = np.flip(Temp_range)
Cavity = np.zeros_like(Temp_range)

# lattice size
L = 12
config = classConfiguration.Configuration(1.0, L)
#pbar = tqdm_notebook(Temp_range)
n_cycles = 10000
n_cool = 6000
n_warm = 6000
n_sites = int(L**2*2/3)
for n in range(n_cool):
    lib.hybrid_Monte_Carlo(config,n_sites,1./0.3)
for i, T in enumerate(Temp_range):
    #pbar.set_description(f"Processing T = {T:.2f}")
    # at each temperature, initialize the Configuration
    av_e, av_e2 = 0, 0

    # Process the Hybrid MC, and get the average value
    for n in range(n_warm):
        lib.hybrid_Monte_Carlo(config,n_sites,1./T)
    for k in range(n_cycles):
        e0 = lib.measure(config,n_sites,1./T)
        av_e += e0
        av_e2 += e0**2

    # normalize averages
    av_e /= float(n_cycles)
    av_e2 /= float(n_cycles)

    # get physical quantities
    Cavity[i] = (av_e2 - av_e**2)/((T**2)*(L**2)*3)
    print(T,Cavity[i])

#Save quantities in a file
np.savetxt("Cavity_%i.dat2"%L, Cavity)

pl.semilogx(Temp_range,Cavity)
pl.show()