import numpy as np
import numpy.random as rnd
import classConfiguration.py

#find the field of one spin sensed which is generated by its nearest neighbors
def near(lattice,N1,N2,n):
    L = lattice.size
    spin = lattice.spin_tensor
    if n==0:
        return spin[N1][N2][1]+spin[N1][N2][2]+spin[N1][(N2+1)%L][2]+spin[(N1-1)%L][(N2+1)%L][1]
    elif n==1:
        return spin[N1][N2][0]+spin[N1][N2][2]+spin[(N1+1)%L][(N2-1)%L][0]+spin[(N1+1)%L][N2][2]
    elif n==2:
        return spin[N1][N2][0]+spin[N1][N2][1]+spin[N1][(N2-1)%L][0]+spin[(N1-1)%L][N2][1]


def CanoMC(kagome):
    L = kagome.size
    beta = kagome.beta
    T = 1./beta
    spin = kagome.spin_tensor
    J = kagome.J
    for n1 in range(L):
        for n2 in range(L):
            for site in range(3):
                delta =rnd.random([3])*2-1
                z = delta[2]*T+spin[n1][n2][2]
                x = delta[0]/np.sqrt((delta[0]**2)+delta[1]**2)*np.sqrt(1-z**2)
                y = delta[1]/np.sqrt((delta[0]**2)+delta[1]**2)*np.sqrt(1-z**2)
                #random change of spin with constriant that delta_Sz < T
                dE = J*np.dot((np.array([x,y,z])-spin[n1][n2][site]),near(kagome,n1,n2,site))
                if rnd.random()<min(1,np.exp(-beta*dE)):
                    kagome.spin_tensor[n1][n2][site]=np.array([x,y,z])
                    kagome.energy += dE
