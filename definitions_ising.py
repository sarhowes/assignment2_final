"""
    ====== Class script for running the simulations ======
              created by: Sarah Howes, Lars Reems
    ======================================================

Includes all the physics formulas that are used in calculating
the Monte Carlo simulation of the Ising Model.

This script is imported into run_code_ising.py in order to perform
all simulations. 

"""

import numpy as np
import copy
import random
from tqdm import tqdm

class Ising_Model:

    def __init__(self,N:float, J:float, ext_mag:float, t_max:float):
        """Initialization of Ising_Model class. Define constant values for computation

        Args:
            N (float): length of grid of spinors. Default is 50
            J (float): value of coupling constant. Default is 1
            ext_mag (float): value of exernal magnetic field. Default is 0
            t_max (float): Maximum time the computation runs after equilibrium.
        """
        self.N = N
        self.J = J
        self.ext_mag = ext_mag
        self.t_max = t_max
        
    def make_grid(self):
        """Make a grid of NxN spinors each in a random 
        direction (up +1 or down -1)

        Returns:
            np.ndarray (shape NxN): array of 1 or -1 with shape (N,N)
        """
        N = self.N
        grid = np.random.randint(0,2,(N,N))
        grid[grid==0] = -1
        return grid
    

    def calc_H(self,grid:np.ndarray, i:int, j:int):
        """Calculate first term of the Hamiltonian for one spinor using four nearest neighbors (left, right, above, below)

        Args:
            grid (np.ndarray): grid of spinors with shape (N,N)
            i (int): row of central spinor
            j (int): column of central spinor

        Returns:
            float: Hamiltonian of one spinor with index (i,j)
        """
        N = self.N
        J = self.J
        center_spinor = grid[i][j]
        spin_sum = center_spinor*(grid[(i+1)%N][j] + grid[i-1][j] + grid[i][(j+1)%N] + grid[i][j-1])
        hamiltonian = -J*spin_sum
        return hamiltonian


    def calc_H_total(self,grid:np.ndarray):
        """Calculate the total Hamiltonian for the full grid of size (N,N)

        Args:
            grid (np.ndarray of shape (N,N)): grid of spinors with shape (N,N)

        Returns:
            float: Hamiltonian of whole grid
        """
        ext_mag = self.ext_mag
        H = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                H += self.calc_H(grid,i,j)

        total_hamiltonian = int(H)-ext_mag*np.sum(grid)

        return total_hamiltonian
    
    
    def calc_p(self,beta:float):
        """Calculate probability of accepting the spin flip if the energy is raised

        Args:
            beta (float): constant dependent on temperature = 1/kT

        Returns:
            dict: dictionary matching the H_diff values to a probability (between 0 and 1) of accepting the spin flip
        """
        probabilities = []
        for H_diff in [1, 2, 3, 4, 5, 6, 7, 8,9, 10]:
            p = np.exp(-beta*(H_diff))
            probabilities.append(p)
        prob_dict = dict(zip([1, 2, 3, 4, 5, 6, 7, 8,9, 10], probabilities))
        return prob_dict



    def update_grid(self, grid:np.ndarray, prob_dict:dict, energy:np.array, magnetization:np.array):
        """Updates grid, magnetization, and energy values given one random spin flip

        Args:
            grid (np.ndarray): 2D array with shape (N,N) of +1 or -1 values
            prob_dict (dict): dictionary that maps the possible energy difference values (4 or 8) to the respective probabilities
            energy (np.array): array of energy values, use last value from array to generate updated energy
            magnetization (np.array): array of magnetization values, use last value from array to generate updated magnetization

        Returns:
            np.ndarray, float, float: updated grid of shape (N,N), new total Hamiltonian, new total magnetization
        """
        N = self.N
        ext_mag = self.ext_mag
        grid_new = copy.deepcopy(grid)
        i,j = np.random.randint(0,N,2)
        grid_new[i][j] *= -1

        H_current = self.calc_H(grid,i,j) - ext_mag*np.sum(grid)
        H_new = self.calc_H(grid_new,i,j) - ext_mag*np.sum(grid_new)

        H_diff = H_new - H_current

        # always accept flip
        if H_diff <= 0:
            new_energy = energy[-1] + 2*H_diff # add H_diff for central and neighboring spins
            new_magnetization = magnetization[-1] + 2*grid_new[i][j]
        else:
            probability_to_change = prob_dict[H_diff]
            # accept with the chance of p
            if probability_to_change > random.random():
                new_energy = energy[-1] + 2*H_diff 
                new_magnetization = magnetization[-1] + 2*grid_new[i][j]
            else:
                # stay in same position if probability not high enough
                new_energy = energy[-1]
                new_magnetization = magnetization[-1]
                grid_new = grid

        return grid_new, new_energy, new_magnetization
    


    def autocorrelation_function(self, mag_eq:np.array):
        """Calculate the autocorrelation function for a system after equilibrium

        Args:
            mag_eq (np.array): magnetization for each time step after equilibrium has been reached

        Returns:
            np.array: autocorrelation function for each time step
        """
        print('FINDING AUTO-CORRELATION FUNCTION...')
        t_max = self.t_max
        mag_eq = mag_eq[1:]
        acf = []
        for t in tqdm(range(t_max)):

            t_diff = t_max - t

            term1 = np.sum(mag_eq[:t_diff]*mag_eq[t:], dtype='int64')
            term2 = np.sum(mag_eq[:t_diff], dtype='int64')
            term3 = np.sum(mag_eq[t:], dtype='int64')

            mult_term = 1/(t_diff)
            acf_total = mult_term*term1 - ((mult_term*term2)*(mult_term*term3))
            acf.append(acf_total)

        acf = np.array(acf)

        return np.array(acf)
    


    def calc_simple_errors(self, tau:float, t_max:float, magnetization_per_spin:np.array, energy_per_spin:np.array):
        """Calculate the errors for the magnetization and energy values after equilibrium

        Args:
            tau (float): the correlation time
            t_max (float): the maximum time after equilibrium the system runs
            magnetization_per_spin (np.array): the magnetization values over time after equilibrium, length t_max
            energy_per_spin (np.array): the energy values over time after equilibrium, length t_max

        Returns:
            float, float: the standard deviation (error) values for the energy and magnetization of the grid
        """
        N = self.N
        sigma_magnetization_per_spin = np.sqrt(2*(tau/t_max) * (np.mean(magnetization_per_spin**2)-np.mean(magnetization_per_spin)**2))
        sigma_energy_per_spin = np.sqrt(2*(tau/t_max) * (np.mean(energy_per_spin**2)-np.mean(energy_per_spin)**2))

        return sigma_magnetization_per_spin, sigma_energy_per_spin
    


    def blocking_method(self, beta:float, temp:float, magnetization_per_spin:np.array, energy_per_spin:np.array):
        """Calculate the magnetic susceptibility and specific heat using the blocking method

        Args:
            beta (float): =1/(k_B T) exponential value dependent on temperature
            temp (float): temperature of the system
            magnetization_per_spin (np.array): magnetization values of one block
            energy_per_spin (np.array): energy values of one block

        Returns:
            float, float: magnetic susceptibility and specific heat values for one block
        """
        N = self.N
        # find the magnetic susceptibility and specific heat using blocking method
        magnetic_susceptibility_per_spin = (beta/N**2) * (np.mean(magnetization_per_spin**2)-np.mean(magnetization_per_spin)**2)
        specific_heat_per_spin = beta/(N**2 * temp) * (np.mean(energy_per_spin**2)-np.mean(energy_per_spin)**2)

        return magnetic_susceptibility_per_spin, specific_heat_per_spin