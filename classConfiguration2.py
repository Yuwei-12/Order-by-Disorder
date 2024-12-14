import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from numba import njit   #used to speed up

class Configuration:
    def __init__(self, J, L):
        self.size = L
        self.J = J
        self.spins = self._init_spins(L)
        self.energy = self._get_energy()
        # print(self.energy)
    
    def _init_spins(self,L):
        # Generate the spin tensor (L × L × 3 × 3)
        spin_tensor = np.zeros((L, L, 3, 3))
    
        #  Every triangular plaquete has three lattice points, each spin of these points is a vector with three dimensions
        for cell_x in range(L):
            for cell_y in range(L):
                for site in range(3):
                    if site == 0: a = 1
                    if site == 1: a = 2
                    if site == 2: a = 0
                    x= 4*cell_x+2*cell_y+a
                    if x%3 == 1:
                        x_component = 0
                        y_component = 1
                        z_component = 0
                    if x%3 == 0:
                        x_component = -np.sqrt(3)/2
                        y_component = -0.5
                        z_component = 0
                    if x%3 == 2:
                        x_component = np.sqrt(3)/2
                        y_component = -0.5
                        z_component = 0
                    # attribute values of the x,y,z-component to the spin_tensor for every triangular plaquette and every lattice point
                    spin_tensor[cell_x, cell_y, site] = [x_component, y_component, z_component]
                
        # Save the generated spin tensor and calculate the hamiltonian  
        return spin_tensor      
        
    def _get_energy(self):
        """Returns the total energy"""
        energy = 0.
        for cell_x in range(self.size):
            for cell_y in range(self.size):
                # Calculate the three components of the sum of the spins in every triangular plaquette
                cell_spin = np.zeros(3)
                cell_spin[0] = np.sum(self.spins[cell_x][cell_y][:,0])
                cell_spin[1] = np.sum(self.spins[cell_x][cell_y][:,1])
                cell_spin[2] = np.sum(self.spins[cell_x][cell_y][:,2])
                cell_spin_2 = np.zeros(3)
                cell_spin_2[0] = self.spins[(cell_x+1)%(self.size)][cell_y %(self.size)][0][0]+self.spins[(cell_x)%(self.size)][(cell_y+1)%(self.size)][1][0]+self.spins[(cell_x+1)%(self.size)][(cell_y+1)%(self.size)][2][0]
                cell_spin_2[1] = self.spins[(cell_x+1)%(self.size)][cell_y %(self.size)][0][1]+self.spins[(cell_x)%(self.size)][(cell_y+1)%(self.size)][1][1]+self.spins[(cell_x+1)%(self.size)][(cell_y+1)%(self.size)][2][1]
                cell_spin_2[2] = self.spins[(cell_x+1)%(self.size)][cell_y %(self.size)][0][2]+self.spins[(cell_x)%(self.size)][(cell_y+1)%(self.size)][1][2]+self.spins[(cell_x+1)%(self.size)][(cell_y+1)%(self.size)][2][2]
                # calculate the spin hamiltonian for this kagome lattice system 
                energy += self.J/2 * (cell_spin[0]**2 + cell_spin[1]**2 + cell_spin[2]**2 + cell_spin_2[0]**2 + cell_spin_2[1]**2 + cell_spin_2[2]**2)
    
        return energy
