import numpy as np
import lib
from tqdm import tqdm
from numba import njit

def T_i_x_y_z(lattice):
    L = lattice.size
    T_i = np.zeros((L,L,3,3,3,3))
    for i in range(L):
        for j in range(L):
            for k in range(3):
                S_spin = lattice.spins[i][j][k]
                # we use 0 stands for x, 1 stands for y, 2 stands for z
                xxy = S_spin[0]*S_spin[0]*S_spin[1]-0.2*S_spin[1]
                xxz = S_spin[0]*S_spin[0]*S_spin[1]-0.2*S_spin[2]
                yyx = S_spin[1]*S_spin[0]*S_spin[1]-0.2*S_spin[0]
                yyz = S_spin[1]*S_spin[2]*S_spin[1]-0.2*S_spin[2]
                zzy = S_spin[2]*S_spin[2]*S_spin[1]-0.2*S_spin[1]
                zzx = S_spin[2]*S_spin[0]*S_spin[2]-0.2*S_spin[0]
                xyz = S_spin[0]*S_spin[1]*S_spin[2]
                T_i[i][j][k][0][0][0] = -yyx-zzx
                T_i[i][j][k][1][1][1] = -xxy-zzy
                T_i[i][j][k][2][2][2] = -yyz-xxz
                T_i[i][j][k][0][1][2] = T_i[i][j][k][0][2][1] = T_i[i][j][k][1][0][2] = T_i[i][j][k][1][2][0]=T_i[i][j][k][2][0][1] = T_i[i][j][k][2][1][0] = xyz
                T_i[i][j][k][0][0][1] = T_i[i][j][k][0][1][0] = T_i[i][j][k][1][0][0] = xxy
                T_i[i][j][k][0][0][2] = T_i[i][j][k][0][2][0] = T_i[i][j][k][2][0][0] = xxz
                T_i[i][j][k][1][1][0] = T_i[i][j][k][1][0][1] = T_i[i][j][k][0][1][1] = yyx
                T_i[i][j][k][1][1][2] = T_i[i][j][k][1][2][1] = T_i[i][j][k][2][1][1] = yyz
                T_i[i][j][k][2][2][1] = T_i[i][j][k][2][1][2] = T_i[i][j][k][1][2][2] = zzy
                T_i[i][j][k][2][2][0] = T_i[i][j][k][2][0][2] = T_i[i][j][k][0][2][2] = zzx
                
    return T_i


# def T_alpha_beta_gamma(kagome,num_sites):
#     N = 20 #number of measurements for one data  
#     for a in range(5):
#         lib.hybrid_Monte_Carlo(kagome,num_sites)
#         #5 intervals to generate a new grid
#     L = kagome.size
#     T_tensor = np.zeros((3, 3, 3))
#     for b in range(N):
#         lib.hybrid_Monte_Carlo(kagome,num_sites)
#         Ti_tensor = T_i_x_y_z(kagome)
#         for alpha in range(3):
#             for beta in range(3):
#                 for gamma in range(3):    
#                     for n_x in range(L):
#                         for n_y in range(L):
#                             for site in range(3):
#                                 T_tensor[alpha, beta, gamma] += Ti_tensor[n_x][n_y][site][alpha][beta][gamma]
#                     T_tensor[alpha, beta, gamma] /= (3 * L**2)
#     #print(T_tensor)
#     return T_tensor/N

def measureChiT(kagome, T, num_sites):
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,1./T)
        #5 intervals to generate a new grid
    L = kagome.size
    chi_T = 0
    lib.hybrid_Monte_Carlo(kagome,num_sites,1./T)
    Tixyz = T_i_x_y_z(kagome)
    for x_i in range(L):
        for y_i in range(L):
            for site_i in range(3):
                for x_j in range(L):
                    for y_j in range(L):
                        for site_j in range(3):
                            for alpha in range(3):
                                for beta in range(3):
                                    for gamma in range(3):
                                        chi_T += Tixyz[x_i][y_i][site_i][alpha][beta][gamma]*Tixyz[x_j][y_j][site_j][alpha][beta][gamma]
    chi_T /= (3 * T * L**2)
    #print(chi_T)
    return chi_T


def measureT2(tensor):
    T2 = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                 T2 += tensor[i][j][k]**2
    return T2


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



def T2S(kagome,num_sites,beta):
    L = kagome.size
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,beta)
        #5 intervals to generate a new grid
    L = kagome.size
    spin = kagome.spins
    Tsqua = T2(spin,L)/((3*L**2)**2)
    return Tsqua