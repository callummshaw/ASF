# ASF
An adaptable modelling framework to model a potential African Swine Fever outbreak in a metapopulation of feral and domestic pigs. A region may consist of several largely independent populations. In turn each population can consist of many groups (in a  hetrogenous configeration) that are connnected via a network, or a single large group (homogeneous config).

The framework consists of three different models, all written in Julia. There is an ODE model, homogenous tau-laeping and a network based hetrogeneous tau-leaping model that includes inter-group dynamics. 

A diagram of meta-population structure is given in below. Where each green region represents a distinct feral population. Within each feral population there are a number of groups of feral pigs, shown in yellow, that interact with each other and select other feral groups. The piggeries/domestic pig populations are shown in red and only interact with select feral groups. 

![ASF Structure](ASF_structure.png?raw=true "ASF Meta-population")

The models are fit using ABC in python (via pyjulia) using pyABC. For speed I created a sysimage of the needed packages to include in the fitting process.
