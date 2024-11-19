import numpy as np

def T_i_x_y_z(lattice):
    L = lattice.size
    T_i = np.zeros((L,L,3,3,3,3))
    for i in range(L):
        for j in range(L):
            for k in range(3):
                S_spin = lattice.spins[i][j][k]
                # we use 0 stands for x, 1 stands for y, 2 stands for z
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            det_s_lm = 0
                            det_s_ln = 0
                            det_s_mn = 0
                            if l == m:
                                det_s_lm = 1/5
                            if l == n:
                                det_s_ln = 1/5
                            if m == n :
                                det_s_mn = 1/5
                            T_i[i][j][k][l][m][n] = S_spin[l]*S_spin[m]*S_spin[n]-S_spin[l]*det_s_mn-S_spin[m]*det_s_ln-S_spin[n]*det_s_lm
    return T_i



