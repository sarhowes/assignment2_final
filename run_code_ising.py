"""
    ====== Main script to run all code ======
          ----------------------------
      created by: Sarah Howes, Lars Reems
    =========================================  

Run this script to perform the main Monte Carlo simulation for the Ising Model.
Using user inputs prompts, you can customize the setup for the simulation.
Or, you can choose the default values for the simulation:
###################################################
* Temperature range: 1.0 - 4.0 in steps of 0.2, 
  including the critical temperature
* No external magnetic field
###################################################

This script imports the class Ising_Model from definitions_ising.py
to run the monte carlo simulation.

This script also will create new folders for each different initial
conditions, labeled with the length of the grid and value for the
external magnetic field. 'results_N_xx_B_ext_xx'

Sub-folders will be created within this folder that stores all the
data that is generated over all the temperature runs, as well as
folders for the figures that are generated.

If you already have data that is stored and want to use this (for
example, just to plot values or run the next section of code without
re-doing the previous section), you can do so by commenting out
whichever part of the run you want to skip (first_run_to_equilibrium,
second_run_after_equilibrium, specific_heat_magnetic_susceptibility)
and un-commenting the np.load() commands for the values that are calculated.

"""


import numpy as np
import matplotlib.pyplot as plt
import random as rm
from tqdm import tqdm
import seaborn as sns
from definitions_ising import Ising_Model
from plotting_definitions_ising import Plotting
import os



def first_run_to_equilibrium(grid:np.ndarray, prob_dict:dict, total_number_steps:int, results_path:str, temp_folder:str):
    """First run over all time steps to reach equilibrium, starting from a random grid

    Args:
        grid (np.ndarray): 2D array of shape NxN with values of either +1 or -1, original randomly generated
        prob_dict (dict): dictionary that maps the possible energy difference values (4 or 8) to the respective probabilities
        total_number_steps (int, optional): total number of time steps taken for the first run. Defaults to total_number_steps
        results_path (str): name of main results folder to save data to
        temp_folder (str): name of temperature folder to save data to

    Returns:
        np.ndarray, np.array, np.array: the updated grid, and the energy and magnetization values per time step
    """
    
    initial_hamiltonian = model.calc_H_total(grid)
    initial_magnetization = np.sum(grid)

    energy = []
    magnetization = []
    energy.append(initial_hamiltonian)
    magnetization.append(initial_magnetization)

    for t_step in tqdm(range(total_number_steps)):

        result = model.update_grid(grid, prob_dict, energy, magnetization)
        grid = result[0]
        energy.append(result[1])
        magnetization.append(result[2])

    energy = np.array(energy)
    magnetization = np.array(magnetization)

    # ## ---------------------------------- save data to folder -----------------------------------------------
    print('Saving first run data (first grid, energy, magnetization arrays)')
    np.save(f'{results_path}/{temp_folder}/first_grid.npy', grid)
    np.save(f'{results_path}/{temp_folder}/first_energy.npy', energy)
    np.save(f'{results_path}/{temp_folder}/first_magnetization.npy', magnetization)
    # ------------------------------------------------------------------------------------------------------

    return grid, energy, magnetization 



def second_run_after_equilibrium(first_grid:np.ndarray, first_energy:np.array, 
                                 first_magnetization:np.array, prob_dict:dict, 
                                 t_max:float, results_path:str, temp_folder:str):
    """Second run of updating the grid, taken after system has reached equilibrium. Uses the grid at equilibrium and
       the final energy and magnetization values after the first run as inputs. This run is used to find the correlation time.

    Args:
        first_grid (np.ndarray): 2D array of shape NxN with values of either +1 or -1, at equilibrium (grid produced after first run)
        first_energy (np.array): energy values generated from the first run, uses the last value from this as the starting point
        first_magnetization (np.array): magnetization values generated from the first run, uses the last value from this as the starting point
        prob_dict (dict): dictionary that maps the possible energy difference values (4 or 8) to the respective probabilities
        t_max (float, optional): total number of time steps to run after equilibrium. Defaults to t_max.
        results_path (str): name of main results folder to save data to
        temp_folder (str): name of temperature folder to save data to

    Returns:
        np.ndarray, np.array, np.array, np.array: the updated grid, energy and magnetization values after equilibrium, 
                                                  and the autocorrelation function over time
    """

    energy_eq = []
    magnetization_eq = []
    energy_eq.append(first_energy[-1])
    magnetization_eq.append(first_magnetization[-1])
    grid = first_grid

    for t_step in tqdm(range(t_max)):
        result_eq = model.update_grid(grid, prob_dict, energy_eq, magnetization_eq)
        grid = result_eq[0]
        energy_eq.append(result_eq[1])
        magnetization_eq.append(result_eq[2])

    energy_eq = np.array(energy_eq)   
    magnetization_eq = np.array(magnetization_eq)
    
    ac_function = model.autocorrelation_function(magnetization_eq)

    if np.any(ac_function < 0) == True:
        first_negative_idx = np.argmax(ac_function<=0)
        ac_function_cut = ac_function[:first_negative_idx]
    else:
        ac_function_cut = ac_function

    tau = np.sum(ac_function_cut/ac_function_cut[0])
    print(f'The correlation time = {tau}')

    # ## ---------------------------------- save data to folder -----------------------------------------------
    print('Saving second run data (second grid, energy magnetization arrays after equilibrium, ACF)')
    np.save(f'{results_path}/{temp_folder}/second_grid.npy', grid)
    np.save(f'{results_path}/{temp_folder}/energy_after_equilibrium.npy', energy_eq)
    np.save(f'{results_path}/{temp_folder}/magnetization_after_equilibrium.npy', magnetization_eq)
    np.save(f'{results_path}/{temp_folder}/autocorrelation_function.npy', ac_function)
    np.save(f'{results_path}/{temp_folder}/tau.npy', tau)
    ## ------------------------------------------------------------------------------------------------------


    return grid, energy_eq, magnetization_eq, ac_function, tau



def specific_heat_magnetic_susceptibility(grid_after_equilibrium:np.ndarray, energy_after_equilibrium:np.array, 
                                          magnetization_after_equilibrium:np.array, tau:float, prob_dict:dict,
                                          results_path:str, temp_folder:str):
    """Calculate the specific heat and magnetic susceptibility using the blocking method. Runs blocks that are of length 16*tau
    and calculates the average and error. 

    Args:
        grid_after_equilibrium (np.ndarray): 2D array of shape NxN with values of either +1 or -1, at equilibrium (grid produced after second run)
        energy_after_equilibrium (np.array): energy values generated from the second run, uses the last value from this as the starting point
        magnetization_after_equilibrium (np.array): magnetization values generated from the second run, uses the last value from this as the starting point
        tau (float): correlation time generated from the second run, used to determine block size
        prob_dict (dict): dictionary that maps the possible energy difference values (4 or 8) to the respective probabilities
        N (float, optional): Grid length in one direction. Defaults to N.
        results_path (str): name of main results folder to save data to
        temp_folder (str): name of temperature folder to save data to

    Returns:
        np.ndarray, (np.array)x4: final grid in equilibrium, magnetization, energy magnetic susceptibility, 
                                  and specific_heat values per block generated from block method
    """

    energy_block = []
    magnetization_block = []
    energy_block.append(energy_after_equilibrium[-1])
    magnetization_block.append(magnetization_after_equilibrium[-1])
    
    magnetic_susceptibility_all_blocks = []
    specific_heat_all_blocks = []
    time_block = int(16*tau)
    grid = grid_after_equilibrium

    for time_step in tqdm(range(10*time_block+1)):
        result_block = model.update_grid(grid, prob_dict, energy_block, magnetization_block)
        grid = result_block[0]
        energy_block.append(result_block[1])
        magnetization_block.append(result_block[2])


        # find values and uncertainties for every block
        if time_step%time_block==0:
            magnetization_per_spin = np.array(magnetization_block)
            energy_per_spin = np.array(energy_block)
            magnetic_susceptibility_per_spin, specific_heat_per_spin = model.blocking_method(beta, temp, magnetization_per_spin[-time_block:], 
                                                                                             energy_per_spin[-time_block:])
            magnetic_susceptibility_all_blocks.append(magnetic_susceptibility_per_spin)
            specific_heat_all_blocks.append(specific_heat_per_spin)
        
    energy_block = np.array(energy_block)
    magnetization_block = np.array(magnetization_block)

    print(f'Magnetic susceptibility = {np.mean(magnetic_susceptibility_all_blocks)} ± {np.std(magnetic_susceptibility_all_blocks)}')
    print(f'Specfic heat = {np.mean(specific_heat_all_blocks)} ± {np.std(specific_heat_all_blocks)}')

    # # # ---------------------------------- save data to folder -----------------------------------------------
    print('Saving third run data (second grid, energy magnetization arrays after blocking method, magnetic sus, spec heat)')
    np.save(f'{results_path}/{temp_folder}/third_grid.npy', grid)
    np.save(f'{results_path}/{temp_folder}/magnetization_block.npy', magnetization_block)
    np.save(f'{results_path}/{temp_folder}/energy_block.npy', energy_block)
    np.save(f'{results_path}/{temp_folder}/magnetic_susceptibility.npy', magnetic_susceptibility_all_blocks)
    np.save(f'{results_path}/{temp_folder}/specific_heat.npy', specific_heat_all_blocks)
    # # # ------------------------------------------------------------------------------------------------------


    return grid, magnetization_block, energy_block, magnetic_susceptibility_all_blocks, specific_heat_all_blocks



def simulation_setup():
    """Set up user inputs that will customize what each run will produce

    Returns:
        list of floats: returns all the needed variables to set up the simulation, including: 
        the number of time steps to take to reach equilibrium
        the number of time steps to take after equilibrium
        the length of the grid side (N)
        Value for external magnetic field
        Temperatures to compute
    """

    print(
        '===========================================================\n'
        '======== Monte Carlo Simulation of 2D Ising Model =========\n'
        '====== simulation by: Sarah Howes and Lars Reems ==========\n'
        '==========================================================='
        )
    print(
        '===== Default simulation values: =====\n'
        '* Number of steps before equilibrium: 5e6\n'
        '* Number of steps after equilibrium: 1e5\n'
        '* Length of grid side (N): 50\n'
        '* External magnetic field: 0.0\n'
        '* Temperature array: 1.0-4.0 in steps of 0.2\n'
        '======================================'
        )
    custom_setup = input('>> Would you like to customize the simulation setup? (y/n): ')
    if custom_setup == 'y':
        total_number_steps = int(float(input('>> Input number of steps before equilibrium: ')))
        t_max = int(float(input('>> Input number of steps after equilibrium: ')))
        N = int(float(input('>> Input length of grid side (N): ')))
        ext_mag = float(input('>> Input external magnetic field: '))
        start_temp = float(input('>> Input starting temperature: '))
        stop_temp = float(input('>> Input final temperature: '))
        step_temp = float(input('>> Input temperature step size: '))
        temperature = np.arange(start_temp, stop_temp, step_temp)
    elif custom_setup == 'n':
        total_number_steps = int(5e6)        
        t_max = int(100000)
        N = 50
        ext_mag = 0.0
        temperature =  np.arange(1.0, 4.0, 0.2)
    else:
        print('Unknown command')
        exit()
    
    temperature = np.round(temperature, 1)
    temperature = list(temperature)
    temperature.append(2.269)
    temperature.sort()
    temperature = np.array(temperature)
    kb = 1
    J = 1
    T_critical = 2*J/(kb*np.log(1+np.sqrt(2)))

    print(
        '==================================\n'
        '==== Final setup parameters: =====\n'
        f'* Number of steps before equilibrium: {total_number_steps}\n'
        f'* Number of steps after equilibrium: {t_max}\n'
        f'* Length of grid side (N): {N}\n'
        f'* External magnetic field: {ext_mag}\n'
        f'* Temperature array: {temperature}\n'
        '=================================='
    )

    proceed = input('>> Proceed with setup? (y/n):')

    if proceed == 'n':
        exit()
    elif proceed == 'y':
        pass
    else:
        print('Unknown command')
        exit()

    return total_number_steps, t_max, N, ext_mag, temperature, kb, J, T_critical



def folder_generator(N:int, ext_mag:float):
    """Generate folders to store all data and plots, in main folder labelled with length of grid
    and exernal magnetic field values

    Args:
        N (int): length of spin grid side
        ext_mag (float): value of external magnetic field

    Returns:
        str: path name for the main folder to store results
    """
    
    results_path = f'results_N_{N}_B_ext_{ext_mag}'
    if not os.path.exists(results_path): 
        print(f'Creating new folder for all results: {results_path}.')
        os.makedirs(results_path)

    if not os.path.exists(f'{results_path}/autocorrelation_figs'): 
        print('Creating new folder for autocorrelation_figs.')
        os.makedirs(f'{results_path}/autocorrelation_figs')

    if not os.path.exists(f'{results_path}/energy_figs'): 
        print('Creating new folder for energy_figs.')
        os.makedirs(f'{results_path}/energy_figs')

    if not os.path.exists(f'{results_path}/grid_figs'): 
        print('Creating new folder for grid_figs.')
        os.makedirs(f'{results_path}/grid_figs')

    if not os.path.exists(f'{results_path}/magnetization_figs'): 
        print('Creating new folder for magnetization_figs.')
        os.makedirs(f'{results_path}/magnetization_figs')
        
    if not os.path.exists(f'{results_path}/temperature_figs'): 
        print('Creating new folder for temperature_figs.')
        os.makedirs(f'{results_path}/temperature_figs')
    
    if not os.path.exists(f'{results_path}/values_over_temperature'): 
        print('Creating new folder for values_over_temperature.')
        os.makedirs(f'{results_path}/values_over_temperature')
    
    return results_path



def original_grid_generator(results_path:str):
    """Initializing original random grid and saving to main directory 
    
    Args:
        results_path (str): path of the main results folder

    Returns:
        np.ndarray: original random grid of spins, size (N,N) with either +1 or -1 values
    """

    if os.path.exists('original_grid.npy'):
        reuse_grid = input('>> Existing original random grid detected. Re-use this? (y/n): ')
    else:
        reuse_grid = 'n'

    if reuse_grid == 'y':
        original_grid = np.load('original_grid.npy')
    elif reuse_grid == 'n':
        print('Creating new random original grid and plotting.')
        original_grid = model.make_grid()
        np.save('original_grid.npy', original_grid)
        ax = sns.heatmap(original_grid, linewidth=0.5, cbar=False, annot=False, cmap='bwr')
        plt.title(f'Spin Orientation for Original Grid')
        plt.savefig(f'{results_path}/grid_figs/original_grid.png', dpi=300)
        plt.close()
    else:
        print('Unknown command')
        exit()
    return original_grid




if __name__ == "__main__":

    ############################################################
    total_number_steps, t_max, N, ext_mag, temperature, kb, J, T_critical = simulation_setup()
    model = Ising_Model(N,J,ext_mag,t_max)
    results_path = folder_generator(N=N, ext_mag=ext_mag)
    original_grid = original_grid_generator(results_path)
    ############################################################

    # --------------------------------------------------
    corr_time_values = []

    magnetization_values_after_equilibrium = []
    magnetization_errors_after_equilibrium = []

    energy_values_after_equilibrium = []
    energy_errors_after_equilibrium = []

    mag_sus_total = []
    mag_sus_total_error = []

    specific_heat_total = []
    specific_heat_total_error = []
    # -------------------------------------------------

    for temp in temperature:
        print(f'==== TEMP: {temp} ====')
        beta = 1/(kb*temp)
        prob_dict = model.calc_p(beta)
        plotting = Plotting(N=N, temp=temp, T_critical=T_critical, results_path=results_path)

        temp_folder = f'data_for_temp_{temp}'
        if not os.path.exists(f'{results_path}/{temp_folder}'): 
            print(f'Creating new folder for {temp_folder}.')
            os.makedirs(f'{results_path}/{temp_folder}')

        #########################################################################################################
        print('FIRST RUN TO EQUILIBRIUM...')
        first_grid, first_energy, first_magnetization = first_run_to_equilibrium(original_grid, prob_dict, total_number_steps, 
                                                                                 results_path, temp_folder)

        ## ------------------------ load in files to skip second calculation ------------------------------------
        # first_grid = np.load(f'{results_path}/{temp_folder}/first_grid.npy')
        # first_energy = np.load(f'{results_path}/{temp_folder}/first_energy.npy')
        # first_magnetization = np.load(f'{results_path}/{temp_folder}/first_magnetization.npy')
        ## ------------------------------------------------------------------------------------------------------
        
        print('plotting first energy and magnetization over time...')
        plotting.plot_energy_and_magnetization_before_equilibrium(first_magnetization, first_energy)

        # #########################################################################################################
        print('SECOND RUN AFTER EQUILIBRIUM...')
        second_grid, energy_after_equilibrium, \
            magnetization_after_equilibrium, \
            autocorrelation_function, tau = second_run_after_equilibrium(first_grid, first_energy, first_magnetization, 
                                                                        prob_dict, t_max, results_path, temp_folder)

        ## ------------------------ load in files to skip second calculation ------------------------------------
        # autocorrelation_function = np.load(f'{results_path}/{temp_folder}/autocorrelation_function.npy')
        # second_grid = np.load(f'{results_path}/{temp_folder}/second_grid.npy')
        # energy_after_equilibrium = np.load(f'{results_path}/{temp_folder}/energy_after_equilibrium.npy')
        # magnetization_after_equilibrium = np.load(f'{results_path}/{temp_folder}/magnetization_after_equilibrium.npy')
        # tau = np.load(f'{results_path}/{temp_folder}/tau.npy')
        
        ## ------------------------------------------------------------------------------------------------------

        print('plotting autocorrelation function over time...')
        plotting.plot_autocorrelation_function(autocorrelation_function)

        #########################################################################################################
        print('FINDING SPECIFIC HEAT AND MAGNETIC SUSCEPTIBILITY...')
        third_grid, magnetization_block, energy_block, \
            magnetic_susceptibility_all_blocks, \
            specific_heat_all_blocks = specific_heat_magnetic_susceptibility(second_grid, energy_after_equilibrium, 
                                                                            magnetization_after_equilibrium,
                                                                            tau, prob_dict, 
                                                                            results_path, temp_folder)
        
        ## ------------------------ load in files to skip third calculation ------------------------------------
        # third_grid = np.load(f'{results_path}/{temp_folder}/third_grid.npy')
        # magnetization_block = np.load(f'{results_path}/{temp_folder}/magnetization_block.npy')
        # energy_block = np.load(f'{results_path}/{temp_folder}/energy_block.npy')
        # magnetic_susceptibility_all_blocks = np.load(f'{results_path}/{temp_folder}/magnetic_susceptibility.npy')
        # specific_heat_all_blocks = np.load(f'{results_path}/{temp_folder}/specific_heat.npy')
        ## ------------------------------------------------------------------------------------------------------


        # find energy and mag error
        magnetization_error, energy_error = model.calc_simple_errors(tau, t_max, magnetization_block/N**2, energy_block/N**2)


        # # # --------------------------- appending results to temperature array -----------------------------------
        corr_time_values.append(tau)

        magnetization_values_after_equilibrium.append(np.mean(magnetization_block)/N**2)
        energy_values_after_equilibrium.append(np.mean(energy_block)/N**2)
        
        mag_sus_total.append(np.mean(magnetic_susceptibility_all_blocks))
        mag_sus_total_error.append(np.std(magnetic_susceptibility_all_blocks))
        
        specific_heat_total.append(np.mean(specific_heat_all_blocks))
        specific_heat_total_error.append(np.std(specific_heat_all_blocks))

        magnetization_errors_after_equilibrium.append(magnetization_error)
        energy_errors_after_equilibrium.append(energy_error)
        # # ------------------------------------------------------------------------------------------------------

        print('plotting final grid...')
        plotting.plot_grid_after_equilibrium(third_grid)


        ################################### END OF TEMPERATURE LOOP #############################################

    # ------------------ list->array and save values from temp loop --------------------------------
    mag_sus_total = np.array(mag_sus_total)
    mag_sus_total_error = np.array(mag_sus_total_error)

    specific_heat_total = np.array(specific_heat_total)
    specific_heat_total_error = np.array(specific_heat_total_error)

    magnetization_values_after_equilibrium = np.array(magnetization_values_after_equilibrium)
    magnetization_errors_after_equilibrium = np.array(magnetization_errors_after_equilibrium)

    energy_values_after_equilibrium = np.array(energy_values_after_equilibrium)
    energy_errors_after_equilibrium = np.array(energy_errors_after_equilibrium)

    corr_time_values = np.array(corr_time_values)


    print('Saving values over temperature.')
    np.save(f'{results_path}/values_over_temperature/mag_values_after_eq.npy', magnetization_values_after_equilibrium)
    np.save(f'{results_path}/values_over_temperature/energy_values_after_eq.npy', energy_values_after_equilibrium)
    np.save(f'{results_path}/values_over_temperature/energy_errors_after_equilibrium.npy', energy_errors_after_equilibrium)
    np.save(f'{results_path}/values_over_temperature/magnetization_errors_after_equilibrium.npy', magnetization_errors_after_equilibrium)
    np.save(f'{results_path}/values_over_temperature/corr_time_values.npy', corr_time_values)
    np.save(f'{results_path}/values_over_temperature/mag_sus_total.npy', mag_sus_total)
    np.save(f'{results_path}/values_over_temperature/mag_sus_total_error.npy', mag_sus_total_error)
    np.save(f'{results_path}/values_over_temperature/specific_heat_total.npy', specific_heat_total)
    np.save(f'{results_path}/values_over_temperature/specific_heat_total_error.npy', specific_heat_total_error)
    np.save(f'{results_path}/values_over_temperature/temperature.npy', temperature)
    # ----------------------------------------------------------------------------------------------

    # ------------------------- load in saved arrays to plot ---------------------------------------
    plotting = Plotting(N=N, temp=None, T_critical=T_critical, results_path=results_path)
    mag_sus_total = np.load(f'{results_path}/values_over_temperature/mag_sus_total.npy')
    mag_sus_total_error = np.load(f'{results_path}/values_over_temperature/mag_sus_total_error.npy')

    specific_heat_total = np.load(f'{results_path}/values_over_temperature/specific_heat_total.npy')
    specific_heat_total_error = np.load(f'{results_path}/values_over_temperature/specific_heat_total_error.npy')

    magnetization_values_after_equilibrium = np.load(f'{results_path}/values_over_temperature/mag_values_after_eq.npy')
    magnetization_errors_after_equilibrium = np.load(f'{results_path}/values_over_temperature/magnetization_errors_after_equilibrium.npy')

    energy_values_after_equilibrium = np.load(f'{results_path}/values_over_temperature/energy_values_after_eq.npy')
    energy_errors_after_equilibrium = np.load(f'{results_path}/values_over_temperature/energy_errors_after_equilibrium.npy')

    corr_time_values = np.load(f'{results_path}/values_over_temperature/corr_time_values.npy')
    temperature = np.load(f'{results_path}/values_over_temperature/temperature.npy')
    # -----------------------------------------------------------------------------------------------

    print('Plotting over temperature...')
    plotting.plot_energy_and_magnetization_over_temp(temperature, magnetization_values_after_equilibrium, energy_values_after_equilibrium, 
                                                    magnetization_errors_after_equilibrium, energy_errors_after_equilibrium)
    plotting.plot_correlation_time_over_temp(temperature, corr_time_values)
    plotting.plot_magnetic_susceptibility_over_temp(temperature, mag_sus_total, mag_sus_total_error)
    plotting.plot_specific_heat_over_temp(temperature, specific_heat_total, specific_heat_total_error)
