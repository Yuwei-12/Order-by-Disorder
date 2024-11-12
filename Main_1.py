import class_Configuration.py
import numpy as np
import Measure.py
from tqdm.notebook import tqdm_notebook

# initialize a configuration
Temp_range = np.hstack([np.arange(0.0001,0.01,0.00015),np.arange(0.01,0.1,0.015),np.arange(0.1,0.4,0.1)])
Cavity = np.zeros_like(Temp_range)

# lattice size
L = 10

pbar = tqdm_notebook(Temp_range)
for i, T in enumerate(pbar):
    pbar.set_description(f"Processing T = {T:.2f}")
    # at each temperature, initialize the Configuration
    config = Configuration(T, 1.0, L)
    av_e, av_e2 = 0, 0
    n_cycles = 5*100000
    n_warmup = 5*10000

    # Process the Hybrid MC, and get the average value
    for n in range(n_warmup + n_cycles):
        if n >= n_warmup:
           av_e += measure(config)
           av_e2 = measure(config)**2

    # normalize averages
    av_e /= float(n_cycles)
    av_e2/= float(n_cycles)

    # get physical quantities
    Cavity[i] = (av_e2 - av_e**2)/(T**2)

#Save quantities in a file

np.savetxt("Cavity_%i.dat"%L, Cavity)   


