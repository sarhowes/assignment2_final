

# Computational Physics Assignment 2
# Monte Carlo simulation of 2D Ising Model

There are three main files for this code:
* `definitions_ising.py`
* `plotting_definitions_ising.py`
* `run_code_ising.py`

**`definitions_ising.py`** includes a class called `Ising_Model` that is imported into `run_code_ising.py`.
It has all the physics formulas that are used to
calculate the Monte Carlo simulation of the Ising
Model, including making the grid of spins, calculating
the hamiltonian, and updating the grid according to
MCMC methods.

**`plotting_definitions_ising.py`** includes a class
called `Plotting` that is imported into `run_code_ising.py`. It has all the functions that are used to plot
the figures that we present in our paper, including
plotting values over time and over temperature.


**`run_code_ising.py`** is the main script. It includes a list of definitions that are used to run the three
sections of the simulation by importing the two classes
mentioned above. A few functions also define user 
inputs that are called in the `__main__` that are used in order for the user to customize the setup for the code
(for example, to adjust the length of the simulation
before and after equilibrium, the size of the grid,
whether or not to use an external magnetic field, and the range of temperature values that are calculated
for).

### Running the code
Running the code is very simple: all you have to do is run `run_code_ising.py` and a series of prompts will follow
in the terminal for you to fill out. **It is recommended to follow the default settings the first time you run the code.**

The `__main__` will call all functions defined in `run_code_ising.py` depending on which are enabled from the user inputs, and the 
simulation will run. The simulation will create new folders within the directory of the scripts. The folders are organized
first based off the universal values chosen for the 
simulation (labelled using the grid size and value
for the external magnetic field). Then sub-folders will be created within these to store data for each 
of the temperature values that are performed. Within these sub-folders, all the values that are calculated over the three runs are saved. A data folder will also be made for average quantities plotted over temperature (labelled "values_over_temperature"). Lastly, folders for the plots will be made, including one for the energy, magnetization, grids, autocorrelation function, and plots over temperature (labelled "temperature_figs"). 

A typical directory will look like this:

* run_code_ising.py
* definitions_ising.py
* data_for_temp_1.0
* data_for_temp_1.2
* values_over_temperature
* temperature_figs
* magnetization_figs

(...etc.)

Lastly, if you have data that is already saved for a run and you wish to simply plot results, you can do so by commenting out the functions you want to skip and un-commenting the np.load() commands. This is explained more in detail in the description for `run_code_ising.py`

Its important to note that adding an external magnetic field has only been tested on values of -1.0, 0.0, 0.5, and 1.0, so selecting values outside this range may result in the code failing to run, this is most likely due to a key error in the probability dictionary that is created.
