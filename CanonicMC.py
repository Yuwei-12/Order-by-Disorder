import numpy as np
import numpy.random as rnd
import class_Configuration.py

#find the closest spin site 
def near(lattice,N,n):
    L = lattice.size
    spin = lattice.spin_tensor
    if n==0:
        return spin[N][1,:]+spin[N][2,:]+spin[(N-L)%L][1,:]+spin[(N-L-1)%L][2,:]
    elif n==1:
        return spin[N][0,:]+spin[N][2,:]+spin[(N-1)%L][2,:]+spin[(N+L)%L][0,:]
    elif n==2:
        return spin[N][0,:]+spin[N][1,:]+spin[(N+1)%L][1,:]+spin[(N+L+1)%L][0,:]


def CanoMC(kagome):
    L = kagome.size
    beta = kagome.beta
    T = 1./beta
    spin = kagome.spin_tensor
    J = kagome.J
    for cell in range(L**2):
        for site in range(3):
            z = spin[cell][site,0]+rnd.random()*2*T-T
            x = (rnd.random()*2-1)*np.sqrt(1-z**2)
            y = np.sqrt(1-z**2-x**2)
            #random change of spin with constriant that delta_Sz < T
            dE = J*np.dot((np.array([x,y,z])-spin[cell][site,:]),near(kagome,cell,site))
            if rnd.random()<min(1,np.exp(-beta*dE)):
                kagome.spin_tensor[cell][site,:]=np.array([x,y,z])
                kagome.energy += dE
