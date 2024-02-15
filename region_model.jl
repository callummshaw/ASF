using ASF
using JLD2
using Graphs

#Here is a sample file to demonstrate how the ASF is best run

function simulate_region(ip, net, save)
    """
    Function to run model and save output to desired directory

    ip is the input directory (directory with input parameters, examples provided in Inputs/)
    
    net is the metapopulation structure we will run on, takes Graphs inputs (given some examples below, if 1 population is desired specify in inputs and will ignore this param)
    
    save is the path to the save directory
    """
	
    sim_output = Model_sim(ip, pop_net = net)
    save_object(save*"/out.jdl2",sim_output.u)
    
end

"""
running the model on a metapopulation grid, note the number of populations specified 
in Simulation_Data.csv in input directory must match the number of nodes in metapopulation 
network or model will defualt to running on a line. Furthermore, if wanting to run on
a single population make populations = 1 in the input file and the pop_net input can be ignored.
"""
println("Testing Model!")  
network = grid([2,2]) #running on a simple grid
simulate_region("Inputs/NT/", network, "Dummy/")
