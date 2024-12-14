import numpy as np
import newlib as lib
from tqdm import tqdm
from numba import njit  #used to speed up

# calculate the third rank tensor T_i_alpha_beta_gamma by using the spin of the system
def T_i_x_y_z(lattice):
    L = lattice.size
    T_i = np.zeros((L,L,3,3,3,3))
    delta = np.eye(3)
    for i in range(L):
        for j in range(L):
            for k in range(3):
                    S = lattice.spins[i][j][k]
                    for alpha in range(3):
                        for beta in range(3):
                            for gamma in range(3):
                                T_i[i][j][k][alpha][beta][gamma] = (S[alpha] * S[beta] * S[gamma]
                                                        - (1/5) * S[alpha] * delta[beta, gamma]
                                                        - (1/5) * S[beta] * delta[alpha, gamma]
                                                        - (1/5) * S[gamma] * delta[alpha, beta])                
    return T_i

# the function to measure the octopolar susceptability Chi_T
def measureChiT(kagome, T, num_sites):
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,1./T)
    # 5 intervals to generate a new grid
    
    L = kagome.size
    T_chi = 0
    Tixyz = T_i_x_y_z(kagome)
    
    for x_i in range(L):
        for y_i in range(L):
            for site_i in range(3):
                for x_j in range(L):
                    for y_j in range(L):
                        for site_j in range(3):
                            T_chi += np.sum(Tixyz[x_i][y_i][site_i]*Tixyz[x_j][y_j][site_j])
    T_chi /= (3 * L**2)
    return T_chi

# the function as a part of the measurement about the octuple tensor, using the spin of the system    
@njit
def T2(spin,L):
    T2 = 0
    for x_i in range(L):
        for y_i in range(L):
            for site_i in range(3):
                for x_j in range(L):
                    for y_j in range(L):
                        for site_j in range(3):
                            T2 += np.dot(spin[x_i][y_i][site_i],spin[x_j][y_j][site_j])**3-0.6*np.dot(spin[x_i][y_i][site_i],spin[x_j][y_j][site_j])
    return T2

# the function to measure the octuple tensor
def T2S(kagome,num_sites,beta):
    L = kagome.size
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,beta)
    # 5 intervals to generate a new grid
    
    spin = kagome.spins
    Tsqua = T2(spin,L)/((3*L**2)**2)
    return Tsqua
