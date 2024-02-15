# ASF
An adaptable modelling framework to model a potential African Swine Fever outbreak in a metapopulation of feral pigs/wild boar. A region may consist of several largely independent populations. In turn each population can consist of many groups (in a heterogeneous configuration) that are connected via a network.

A diagram of meta-population structure is given in below. Where each green region represents a distinct feral population. Within each feral population there are a number of groups of feral pigs, shown in yellow, that interact with each other and select other feral groups. The piggeries/domestic pig populations are shown in red and only interact with select feral groups. 

![ASF Structure](ASF_structure.png?raw=true "ASF Meta-population")

region_model.jl is an example file to show how the model is run. 

All model code is in the src directory. 

Model input for different regions is in the Input directory. Each subdirectory is for a different region. The input directory contains a Simulation_Data.csv file that contains the meta-data for the simulation, a Population.csv file that contains population specific parameters and a Seasonal.csv file that contains parameters that can vary seasonally. 
