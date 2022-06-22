# ASF
Adapative modelling frame work to model a potential African Swine Fever outbreak in a metapopulation of feral and domestic pigs within Australia.

Model is written in Julia and uses tau-leaping. A diagram of meta-population structure is given in below. Where each green region represents a distinct feral population. Within each feral population there are a number of groups of feral pigs, shown in yellow, that interact with each other and select other feral groups. The piggeries/domestic pig populations are shown in red and only interact with select feral groups. 

![ASF Structure](ASF_structure.png?raw=true "ASF Meta-population")
