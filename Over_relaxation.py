import numpy as np
import numpy.random as rnd
import class_Configuration.py
import Canonic.MC.py

L = lattice.size
num_sites = L*L*2/3 # The number of sites which are chosen
# choose random sites in the lattice
def choose_random_sites(lattice, num_sites):
    L = lattice.size
    #Generate all possible (x , y) coordinates in an L*L lattice
    # n is the label of spins in one Triangle, and in the range: 0, 1, 2
    all_sites = [(x, y, n) for x in range(L) for y in range(L) for n in range([0, 1, 2])]

    # Randomly select 'num_sites' unique coordinates
    selected_sites = rnd.sample(all_sites, num_sites)
    return selected_sites
# The state of spin after the flip
# former_stete is the coordinates (x,y,n) in the lattice of the choosen spins
def later_state(former_state):
    h = near(former_state)  # to be decided the input of near function
    Bef_flip = Configuration.spin_tensor[former_state[0]][former_state[1]][former_state[2]] # the spin before flip
    h_2 = h[0]**2 +h[1]**2 +h[2]**2
    Si_hi = former_state[0]*h[0] + former_state[1]*h[1] + former_state[2]*h[2]
    Fec_H = np.zeros(3)
    Fec_H = 2*(h*Si_hi/h_2)

    # Aft_flip is the new state of the spin after the operation of flipping
    Aft_flip = np.zeros(3)
    Aft_flip = -Bef_flip + Fec_H
    return Aft_flip

def Over_relaxation(lattice):
    L = lattice.size
    selected_sites = choose_random_sites()
    for i in range (num_sites):
        Cho_site = selected_sites[i]
        Aft_flip = later_state(Cho_site)
        Configuration.spin_tensor[Cho_site[0]][Cho_site[1]][Cho_site[2]] = Aft_flip

  return Configuration.spin_tensor

