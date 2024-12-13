import numpy as np
import newlib2 as lib
from tqdm import tqdm
from numba import njit

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
                    # S_spin = lattice.spins[i][j][k]
                # # we use 0 stands for x, 1 stands for y, 2 stands for z
                # xxy = S_spin[0]*S_spin[0]*S_spin[1]-0.2*S_spin[1]
                # xxz = S_spin[0]*S_spin[0]*S_spin[1]-0.2*S_spin[2]
                # yyx = S_spin[1]*S_spin[0]*S_spin[1]-0.2*S_spin[0]
                # yyz = S_spin[1]*S_spin[2]*S_spin[1]-0.2*S_spin[2]
                # zzy = S_spin[2]*S_spin[2]*S_spin[1]-0.2*S_spin[1]
                # zzx = S_spin[2]*S_spin[0]*S_spin[2]-0.2*S_spin[0]
                # xyz = S_spin[0]*S_spin[1]*S_spin[2]
                # T_i[i][j][k][0][0][0] = -yyx-zzx
                # T_i[i][j][k][1][1][1] = -xxy-zzy
                # T_i[i][j][k][2][2][2] = -yyz-xxz
                # T_i[i][j][k][0][1][2] = T_i[i][j][k][0][2][1] = T_i[i][j][k][1][0][2] = T_i[i][j][k][1][2][0]=T_i[i][j][k][2][0][1] = T_i[i][j][k][2][1][0] = xyz
                # T_i[i][j][k][0][0][1] = T_i[i][j][k][0][1][0] = T_i[i][j][k][1][0][0] = xxy
                # T_i[i][j][k][0][0][2] = T_i[i][j][k][0][2][0] = T_i[i][j][k][2][0][0] = xxz
                # T_i[i][j][k][1][1][0] = T_i[i][j][k][1][0][1] = T_i[i][j][k][0][1][1] = yyx
                # T_i[i][j][k][1][1][2] = T_i[i][j][k][1][2][1] = T_i[i][j][k][2][1][1] = yyz
                # T_i[i][j][k][2][2][1] = T_i[i][j][k][2][1][2] = T_i[i][j][k][1][2][2] = zzy
                # T_i[i][j][k][2][2][0] = T_i[i][j][k][2][0][2] = T_i[i][j][k][0][2][2] = zzx
                
    return T_i


def T_alpha_beta_gamma(kagome):
    L = kagome.size
    T_tensor = np.zeros((3, 3, 3))
    Ti_tensor = T_i_x_y_z(kagome)  
    for n_x in range(L):
        for n_y in range(L):
            for site in range(3):
                T_tensor += Ti_tensor[n_x][n_y][site]
    T_tensor /= (3 * L**2)
    return T_tensor

def measureChiT(kagome, T, num_sites):
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,1./T)
    #     #5 intervals to generate a new grid
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


def measureT2(kagome,num_sites,T):
    T2 = 0
    for a in range(5):
        lib.hybrid_Monte_Carlo(kagome,num_sites,1./T)
    tensor = T_alpha_beta_gamma(kagome)
    T2 = np.sum(tensor*tensor)
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