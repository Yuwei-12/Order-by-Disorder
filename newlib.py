import numpy as np
import numpy.random as rnd
from numba import njit


@njit
#find the field of one spin sensed which is generated by its nearest neighbors
def near(L,spin,N1,N2,n):
    if n==0:
        return spin[N1][N2][1]+spin[N1][N2][2]+spin[N1][(N2+1)%L][2]+spin[(N1-1)%L][(N2+1)%L][1]
    elif n==1:
        return spin[N1][N2][0]+spin[N1][N2][2]+spin[(N1+1)%L][(N2-1)%L][0]+spin[(N1+1)%L][N2][2]
    elif n==2:
        return spin[N1][N2][0]+spin[N1][N2][1]+spin[N1][(N2-1)%L][0]+spin[(N1-1)%L][N2][1]

@njit
def CanoMC(L,beta,J,spin):
    T = 1./beta
    E = 0
    for n1 in range(L):
        for n2 in range(L):
            for site in range(3):
                delta =rnd.random((3,))
                z = 1-T + delta[2]*T
                x = delta[0]/np.sqrt(delta[0]**2+delta[1]**2)*np.sqrt(1-z**2)
                y = delta[1]/np.sqrt(delta[0]**2+delta[1]**2)*np.sqrt(1-z**2)
                normal = spin[n1][n2][site]
                if normal[0] != 0 or normal[1] != 0:
                    tangent1 = np.array([-normal[1], normal[0], 0], dtype=np.float64)
                else:
                    tangent1 = np.array([1, 0, 0],dtype=np.float64)
                tangent1 /= np.linalg.norm(tangent1)
                tangent2 = np.cross(normal, tangent1)
                tangent2 /= np.linalg.norm(tangent2)
                new_S = z*normal+x*tangent1+y*tangent2
                dE = J*np.dot((new_S-spin[n1][n2][site]),near(L,spin,n1,n2,site))
                if dE<0:
                    spin[n1][n2][site]=new_S
                    E += dE
                elif rnd.random()<min(1,np.exp(-beta*dE)):
                    spin[n1][n2][site]=new_S
                    E += dE
    return spin,E

def choose_random_sites(lattice, num_sites):
    L = lattice.size
    #Generate all possible (x , y) coordinates in an L*L lattice
    # n is the label of spins in one Triangle, and in the range: 0, 1, 2
    selected_sites = []
    for i in range(num_sites):
        x = rnd.randint(0,L)
        y = rnd.randint(0,L)
        n = rnd.randint(0,3)
        selected_sites.append([x,y,n])
    #all_sites = [(x, y, n) for x in range(L) for y in range(L) for n in range(3)]
    # Randomly select 'num_sites' unique coordinates
    #selected_sites = rnd.sample(all_sites, num_sites)
    return selected_sites
# The state of spin after the flip
# former_stete is the coordinates (x,y,n) in the lattice of the choosen spins
def later_state(lattice,former_state):
    h = near(lattice.size,lattice.spins,former_state[0],former_state[1],former_state[2])  # to be decided the input of near function
    Bef_flip = lattice.spins[former_state[0]][former_state[1]][former_state[2]] # the spin before flip
    h_2 = h[0]**2 +h[1]**2 +h[2]**2
    Si_hi = Bef_flip[0]*h[0] + Bef_flip[1]*h[1] + Bef_flip[2]*h[2]
    Fec_H = np.zeros(3)
    Fec_H = 2*(h*Si_hi/h_2)

    # Aft_flip is the new state of the spin after the operation of flipping
    Aft_flip = -Bef_flip + Fec_H
    #Aft_flip = Aft_flip/np.sqrt(Aft_flip[0]**2+Aft_flip[1]**2+Aft_flip[2]**2)
    return Aft_flip

def Over_relaxation(lattice,num_sites):
    L = lattice.size
    selected_sites = choose_random_sites(lattice,num_sites)
    for i in range (num_sites):
        Cho_site = selected_sites[i]
        Aft_flip = later_state(lattice,Cho_site)
        lattice.spins[Cho_site[0]][Cho_site[1]][Cho_site[2]] = Aft_flip


def hybrid_Monte_Carlo(kagome,num_sites,beta): 
    num_of_over_relax = 4
    spin , dE = CanoMC(kagome.size,beta,kagome.J,kagome.spins)
    kagome.spins = spin
    kagome.energy += dE
    for k in range(num_of_over_relax):
        Over_relaxation(kagome,num_sites)

def measure(kagome,num_sites,beta):  
    for a in range(5):
        hybrid_Monte_Carlo(kagome,num_sites,beta)
        #5 intervals to generate a new grid
    return kagome.energy