# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:03:48 2024

@author: Lenovo
"""
import numpy as np
import lib_capacity

def T_alpha_beta_gamma(kagome,num_sites):
    N = 20 #number of measurements for one data  
    for a in range(5):
        lib_capacity.hybrid_Monte_Carlo(kagome,num_sites)
        #5 intervals to generate a new grid
    L = kagome.size
    T_tensor = np.zeros(3, 3, 3)
    
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                for b in range(N):
                    lib_capacity.hybrid_Monte_Carlo(kagome,num_sites)
                    for n_x in range(L):
                        for n_y in range(L):
                            for site in range(3):
                                T_tensor[alpha, beta, gamma] += T_i_x_y_z[n_x, n_y, site, alpha, beta, gamma]
                T_tensor[alpha, beta, gamma] /= (3 * N * L**2)
    print(T_tensor)
    return T_tensor

def measure_chi_T(kagome, T, num_sites):
    N = 20 #number of measurements for one data  
    for a in range(5):
        lib_capacity.hybrid_Monte_Carlo(kagome,num_sites)
        #5 intervals to generate a new grid
    L = kagome.size
    chi_T = 0
    
    for b in range(N):
        lib_capacity.hybrid_Monte_Carlo(kagome,num_sites)
        for x_i in range(L):
            for y_i in range(L):
                for site_i in range(3):
                    for x_j in range(L):
                        for y_j in range(L):
                            for site_j in range(3):
                                for alpha in range(3):
                                    for beta in range(3):
                                        for gamma in range(3):
                                            chi_T += T_i_x_y_z[x_i, y_i, site_i, alpha, beta, gamma]*T_i_x_y_z[x_j, y_j, site_j, alpha, beta, gamma]
    chi_T /= (3 * N * T * L**2)
    print(chi_T)
    return chi_T
