import numpy as np
from tqdm import tqdm
import lib
import classConfiguration


# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.01,0.00033),np.arange(0.0105,0.1,0.015),np.arange(0.13,0.4,0.1)])
Cavity = np.zeros_like(Temp_range)

# lattice size
L = 12

#pbar = tqdm_notebook(Temp_range)
for i, T in tqdm(enumerate(Temp_range)):
    #pbar.set_description(f"Processing T = {T:.2f}")
    # at each temperature, initialize the Configuration
    config = classConfiguration.Configuration(T, 1.0, L)
    av_e, av_e2 = 0, 0
    n_cycles = 5*10000
    n_warmup = 5*10000
    n_sites = int(L**2/3)
    # Process the Hybrid MC, and get the average value
    for n in range(n_warmup):
        num_of_over_relax = 4
        lib.hybrid_Monte_Carlo(config,n_sites)
    for k in range(n_cycles):
           av_e += lib.measure(config,n_sites)
           av_e2 += lib.measure(config,n_sites)**2

    # normalize averages
    av_e /= float(n_cycles)
    av_e2/= float(n_cycles)

    # get physical quantities
    Cavity[i] = (av_e2 - av_e**2)/((T**2)*(L**2)*3)
    print(av_e**2,av_e2,Cavity[i])

#Save quantities in a file

np.savetxt("Cavity_%i.dat"%L, Cavity)


