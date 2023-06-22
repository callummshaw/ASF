using ASF
using JLD2
using DifferentialEquations
using Graphs

function simulate_region(ip, net, sf)
	
    save = "/home/callum/Desktop/ASFAUS/Line16/"
    
    if !isdir(save*sf)
        mkdir(save*sf)
    end



    μfy = [0.5]

    for i in μfy
        println(i)
        sn = Int(i*10)
        sim_output = Model_sim(ip, adj = i)
        save_object(save*sf*"/out$(sn).jdl2",sim_output.u)
    end
end

network = ladder_graph(2)

println("ACT")
simulate_region("Inputs/ACT/",network,"ACT")
#println("NSW")
#simulate_region("Inputs/NSW/",network, "NSW")
#println("QLD")
#simulate_region("Inputs/QLD/",network, "QLD")
#println("NT")
#simulate_region("Inputs/NT/",network, "NT")
