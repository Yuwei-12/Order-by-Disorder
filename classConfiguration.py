import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

class Configuration:
    

    def __init__(self, T, J, L):
        self.size = L
        self.J = J
        self.beta = 1./T
        # Generate the spin tensor (L × L × 3 × 3)
        spin_tensor = np.zeros((L, L, 3, 3))
    
        #  Every triangular plaquete has three lattice points, each spin of these points is a vector with three dimensions
        for cell_x in range(L):
            for cell_y in range(L):
                for site in range(3):
                
                    # Generate the spin of every lattice point randomly
                    x_component = np.random.uniform(-1, 1)
                    y_component = np.random.uniform(-1, 1)
                    z_component = np.random.uniform(-1, 1)
                    norm = np.sqrt(x_component**2 + y_component**2 + z_component**2)
                    x_component /= norm
                    y_component /= norm
                    z_component /= norm
                    # attribute values of the x,y,z-component to the spin_tensor for every triangular plaquette and every lattice point
                    spin_tensor[cell_x, cell_y, site] = [x_component, y_component, z_component]
                
        # Save the generated spin tensor and calculate the hamiltonian        
        
        self.spins = spin_tensor
        self.energy = self._get_energy()
        
        
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
