# ASF
An adaptable modelling framework to model a potential African Swine Fever outbreak in a metapopulation of feral and domestic pigs.

Base model is written in Julia and uses tau-leaping. This model is highly heterogeneous as it runs on a network and includes inter-group dynamics. There are also faster ODE and SDE models that do not consider groups and models each region as one homogenous population. 

A diagram of meta-population structure is given in below. Where each green region represents a distinct feral population. Within each feral population there are a number of groups of feral pigs, shown in yellow, that interact with each other and select other feral groups. The piggeries/domestic pig populations are shown in red and only interact with select feral groups. 

![ASF Structure](ASF_structure.png?raw=true "ASF Meta-population")
