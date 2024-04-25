"""
    ====== Class script for plotting results ======
              created by: Sarah Howes, Lars Reems
    ======================================================

Includes functions that will plot the results generated from
the Monte Carlo simulation of the Ising Model.

This script is imported into run_code_3D.py in order to plot
all the results, including plots over time, over temperature,
and the final positions of the spin grid.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plotting:

    def __init__(self,N,temp,T_critical,results_path):
        self.N = N
        self.temp = temp
        self.T_critical = T_critical
        self.results_path = results_path



    # plot over time
    def plot_energy_and_magnetization_before_equilibrium(self, first_magnetization:np.array, first_energy:np.array):
        """Plot the energy and magnetization values per spin over time sweeps, including values before equilibrium was reached.
        Uses values attained from the first run (first_run_to_equilibrium). Saves plot to folders energy_figs and
        magnetization_figs.

        Args:
            first_magnetization (np.array): magnetization values attained from the first run to equilibrium
            first_energy (np.array): energy values attained from the first run to equilibrium
        """

        N = self.N
        temp = self.temp
        results_path = self.results_path
        time_sweeps = np.array( range( len(first_magnetization) ) )/(N**2)

        plt.plot(time_sweeps, first_magnetization/(N**2), color='darkblue')
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time Step [sweeps]')
        plt.ylabel('Average Spin')
        plt.title(f'$T={temp}$')
        plt.savefig(f'{results_path}/magnetization_figs/mag_temp_{temp}.png', dpi=300)
        plt.close()

        plt.plot(time_sweeps, first_energy/(N**2), color='darkblue')
        plt.ylim(-4.1, 0.1)
        plt.title(f'$T={temp}$')
        plt.xlabel('Time Step [sweeps]')
        plt.ylabel('Energy')
        plt.savefig(f'{results_path}/energy_figs/energy_temp_{temp}.png', dpi=300)
        plt.close()



    def plot_autocorrelation_function(self, autocorrelation_function:np.array):
        """Plot the autocorrelation function of the grid of spinors over time after equilibrium, normalized to the initial value.
        Includes a exponential curve that is fitted to the values before zero is reached.
        Saves the plot in the folder autocorrelation_figs.

        Args:
            autocorrelation_function (np.array): autocorrelation values over time after equilibrium, length is t_max

        """
        N = self.N
        temp = self.temp
        results_path = self.results_path
        time_sweeps = np.array( range(len(autocorrelation_function)) )/(N**2)

        acf = autocorrelation_function/autocorrelation_function[0]

        # curve fitting
        def exp_fcn(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        from scipy.optimize import curve_fit

        if np.any(acf < 0) == True:
            first_negative_idx = np.argmax(acf<=0)
            xdata = time_sweeps[:first_negative_idx]
            ydata = acf[:first_negative_idx]
        else:
            xdata = time_sweeps
            ydata = acf

        popt, pcov = curve_fit(exp_fcn, xdata, ydata, maxfev=5000)

        plt.plot(time_sweeps, acf, color='darkblue')
        plt.ylim(-1.1, 1.3)
        plt.plot(xdata, exp_fcn(xdata, *popt), ls='--', color='violet', label='fit $ae^{-bx}+c$: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.axhline(0, ls='--', color='k')
        plt.legend()
        plt.title(f'$T={temp}$')
        plt.xlabel('Time Step [sweeps]')
        plt.ylabel('$\chi$(t)/$\chi$(0)')
        plt.savefig(f'{results_path}/autocorrelation_figs/ac_temp_{temp}.png', dpi=300)
        plt.close()



    def plot_grid_after_equilibrium(self, third_grid:np.ndarray):
        """Plot the final grid after equilibrium has been reached, with the positive and negative spins represented
        as red and blue tiles on a heatmap, generated after the third run (specific_heat_magnetic_susceptibility).
        Saved to the folder grid_figs.

        Args:
            third_grid (np.ndarray): 2D array of shape NxN with values of either +1 or -1, at equilibrium (grid produced after third run)
        """
        temp = self.temp
        results_path = self.results_path
        # print(third_grid)
        from matplotlib.colors import ListedColormap
        cmap_dict = {-1: 'blue', 1: 'red'}
        custom_cmap = ListedColormap([cmap_dict[i] for i in [-1,1]])
        ax = sns.heatmap(third_grid, cmap=custom_cmap, linewidth=0.5, cbar=False)
        plt.title(f'Spin Orientation for T = {temp}')
        plt.savefig(f'{results_path}/grid_figs/grid_temp_{temp}.png', dpi=300)
        plt.close()




    # plot over temperature
    def plot_energy_and_magnetization_over_temp(self, temperature:np.array, magnetization_values_after_equilibrium:np.array, 
                                                energy_values_after_equilibrium:np.array, magnetization_errors_after_equilibrium:np.array, 
                                                energy_errors_after_equilibrium:np.array):
        """Plot the average energy and magnetization values and their errors for each temperature. Use the final E, M values after 
        equilibrium has been reached. Saved to folder temperature_figs

        Args:
            temperature (float): array of temperature values to plot
            magnetization_values_after_equilibrium (np.array): array of average magnetization values to plot
            energy_values_after_equilibrium (np.array):  array of average energy values to plot
            magnetization_errors_after_equilibrium (np.array):  array of magnetization errors values to plot
            energy_errors_after_equilibrium (np.array):  array of energy error values to plot
        """

        
        T_critical = self.T_critical
        results_path = self.results_path
        plt.errorbar(temperature, np.abs(magnetization_values_after_equilibrium), 
                     yerr=magnetization_errors_after_equilibrium, color='darkblue', marker='.')
        plt.xlabel('Temperature')
        plt.ylabel('Average Magnetization')
        plt.axvline(T_critical, label='$T_c=2.269$', color='red', ls='--')
        plt.legend()
        plt.savefig(f'{results_path}/temperature_figs/magnetization_over_temp.png', dpi=300)
        plt.close()


        plt.errorbar(temperature, energy_values_after_equilibrium, 
                     yerr=energy_errors_after_equilibrium, color='darkblue', marker='.')
        plt.xlabel('Temperature')
        plt.ylabel('Average Energy')
        plt.axvline(T_critical, label='$T_c=2.269$', color='red', ls='--')
        plt.legend()
        plt.savefig(f'{results_path}/temperature_figs/energy_over_temp.png', dpi=300)
        plt.close()



    def plot_correlation_time_over_temp(self, temperature:np.array, corr_time_values:np.array):
        """Plot the correlation time (tau) values over temperature. Save to folder temperature_figs

        Args:
            temperature (np.array): array of temperature values to plot
            corr_time_values (np.array): array of correlation time (tau) values to plot
        """
        results_path = self.results_path
        T_critical = self.T_critical
        plt.plot(temperature, corr_time_values, marker='.', color='darkblue')
        plt.xlabel('Temperature')
        plt.ylabel('Corelation Time')
        plt.axvline(T_critical, label='$T_c=2.269$', color='red', ls='--')
        plt.legend()
        plt.savefig(f'{results_path}/temperature_figs/correlation_time_over_temp.png', dpi=300)
        plt.close()



    def plot_magnetic_susceptibility_over_temp(self, temperature:np.array, mag_sus_total:np.array, mag_sus_total_error:np.array):
        """Plot the magnetic susceptibility over temperature. Saves to folder temperature_figs

        Args:
            temperature (np.array): array of temperature values to plot
            mag_sus_total (np.array): array of mean magnetic susceptibility values
            mag_sus_total_error (np.array): array of the standard deviation of magnetic susceptibility values
        """
        T_critical = self.T_critical
        results_path = self.results_path
        plt.errorbar(temperature, mag_sus_total, yerr=mag_sus_total_error, marker='.', color='darkblue')
        plt.xlabel('Temperature')
        plt.ylabel('Magnetic Susceptibility')
        plt.axvline(T_critical, label='$T_c=2.269$', color='red', ls='--')
        plt.legend()
        plt.savefig(f'{results_path}/temperature_figs/magnetic_susceptibility_over_temp.png', dpi=300)
        plt.close()



    def plot_specific_heat_over_temp(self, temperature:np.array, specific_heat_total:np.array, specific_heat_total_error:np.array):
        """Plot the specific heat over temperature. Saves to folder temperature_figs

        Args:
            temperature (np.array): array of temperature values to plot
            specific_heat_total (np.array): array of mean specific heat values 
            specific_heat_total_error (np.array): array of the standard deviation of specific heat values
        """
        T_critical = self.T_critical
        results_path = self.results_path
        plt.errorbar(temperature, specific_heat_total, yerr=specific_heat_total_error, marker='.', color='darkblue')
        plt.xlabel('Temperature')
        plt.ylabel('Specific Heat')
        plt.axvline(T_critical, label='$T_c=2.269$', color='red', ls='--')
        plt.legend()
        plt.savefig(f'{results_path}/temperature_figs/specific_heat_over_temp.png', dpi=300)
        plt.close()





