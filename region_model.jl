using ASF
using JLD2
using DifferentialEquations
using Graphs

function simulate_region(ip, net, sf)
	
	save = "/home/callum/Desktop/ACT250/"
    
    if !isdir(save*sf)
        mkdir(save*sf)
    end
    i = 55
    println(i)
   
    sim_output = Model_sim(ip)
    save_object(save*sf*"/out.jdl2",sim_output.u)
   
end

network = ladder_graph(2)

println("ACT")
simulate_region("Inputs/ACT/",network,"ACT")
println("ACT High")
simulate_region("Inputs/ACTH/",network,"ACT_High")
println("Low")
simulate_region("Inputs/ACTL/",network,"ACT_Low")


#println("NSW")
#simulate_region("Inputs/NSW/",network, "NSW")
#println("QLD")
#simulate_region("Inputs/QLD/",network, "QLD")
#println("NT")
#simulate_region("Inputs/NT/",network, "NT")
