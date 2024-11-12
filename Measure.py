import HybridMC.py
import numpy as np

def measure(kagome):
    N = 30 #number of measurements for one data  
    for a in range(5):
        hybridMC()
        #5 intervals to generate a new grid
    E_tem = 0 #temporary average energy
    for b in range(N):
        hybridMC()
        E_tem += kagome.energy/N
    return E_tem

            

        
