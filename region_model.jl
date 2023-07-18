using ASF
using JLD2
using Graphs

function simulate_region(ip, np, net, save)

	fymort = [0.5,0.8,0.95]

    for i in fymort
        println(i)
        if np == 1
            sim_output = Model_sim(ip, fym = i)
        else
            sim_output = Model_sim(ip, pop_net = net, fym = i)
        end
        
        if i == 0.5
            save_object(save*"Low"*"/out.jdl2",sim_output.u)
        elseif i == 0.8
            save_object(save*"Regular"*"/out.jdl2",sim_output.u)
        else
            save_object(save*"High"*"/out.jdl2",sim_output.u)
        end
    end

end


println("ACT")  
network = grid([2,2])
println("250")
simulate_region("Inputs/one250/ACT/",1, network, "/home/callum/Desktop/ASF_Output/ACT/250/One/")

println("1000")
simulate_region("Inputs/one1000/ACT/",1, network, "/home/callum/Desktop/ASF_Output/ACT/1000/One/")

println("4000")
simulate_region("Inputs/one4000/ACT/",1, network, "/home/callum/Desktop/ASF_Output/ACT/4000/One/")

println("4000: 4x1000 Grid")
simulate_region("Inputs/four4000/ACT/",4, network, "/home/callum/Desktop/ASF_Output/ACT/4000/Grid_4/")

println("4000: 4x1000 Line")
network = path_graph(4)
simulate_region("Inputs/four4000/ACT/",4, network, "/home/callum/Desktop/ASF_Output/ACT/4000/Line_4/")

println("4000: 16x250 Grid")
network = grid([4,4])
simulate_region("Inputs/sixteen4000/ACT/",4, network, "/home/callum/Desktop/ASF_Output/ACT/4000/Grid_16/")

println("4000: 16x250 Line")
network = path_graph(16)
simulate_region("Inputs/sixteen4000/ACT/",4, network, "/home/callum/Desktop/ASF_Output/ACT/4000/Line_16/")



println("NSW")  
network = grid([2,2])
println("250")
simulate_region("Inputs/one250/NSW/",1, network, "/home/callum/Desktop/ASF_Output/NSW/250/One/")

println("1000")
simulate_region("Inputs/one1000/NSW/",1, network, "/home/callum/Desktop/ASF_Output/NSW/1000/One/")

println("4000")
simulate_region("Inputs/one4000/NSW/",1, network, "/home/callum/Desktop/ASF_Output/NSW/4000/One/")

println("4000: 4x1000 Grid")
simulate_region("Inputs/four4000/NSW/",4, network, "/home/callum/Desktop/ASF_Output/NSW/4000/Grid_4/")

println("4000: 4x1000 Line")
network = path_graph(4)
simulate_region("Inputs/four4000/NSW/",4, network, "/home/callum/Desktop/ASF_Output/NSW/4000/Line_4/")

println("4000: 16x250 Grid")
network = grid([4,4])
simulate_region("Inputs/sixteen4000/NSW/",4, network, "/home/callum/Desktop/ASF_Output/NSW/4000/Grid_16/")

println("4000: 16x250 Line")
network = path_graph(16)
simulate_region("Inputs/sixteen4000/NSW/",4, network, "/home/callum/Desktop/ASF_Output/NSW/4000/Line_16/")

println("QLD")  
network = grid([2,2])
println("250")
simulate_region("Inputs/one250/QLD/",1, network, "/home/callum/Desktop/ASF_Output/QLD/250/One/")

println("1000")
simulate_region("Inputs/one1000/QLD/",1, network, "/home/callum/Desktop/ASF_Output/QLD/1000/One/")

println("4000")
simulate_region("Inputs/one4000/QLD/",1, network, "/home/callum/Desktop/ASF_Output/QLD/4000/One/")

println("4000: 4x1000 Grid")
simulate_region("Inputs/four4000/QLD/",4, network, "/home/callum/Desktop/ASF_Output/QLD/4000/Grid_4/")

println("4000: 4x1000 Line")
network = path_graph(4)
simulate_region("Inputs/four4000/QLD/",4, network, "/home/callum/Desktop/ASF_Output/QLD/4000/Line_4/")

println("4000: 16x250 Grid")
network = grid([4,4])
simulate_region("Inputs/sixteen4000/QLD/",4, network, "/home/callum/Desktop/ASF_Output/QLD/4000/Grid_16/")

println("4000: 16x250 Line")
network = path_graph(16)
simulate_region("Inputs/sixteen4000/QLD/",4, network, "/home/callum/Desktop/ASF_Output/QLD/4000/Line_16/")



println("NT")  
network = grid([2,2])
println("250")
simulate_region("Inputs/one250/NT/",1, network, "/home/callum/Desktop/ASF_Output/NT/250/One/")

println("1000")
simulate_region("Inputs/one1000/NT/",1, network, "/home/callum/Desktop/ASF_Output/NT/1000/One/")

println("4000")
simulate_region("Inputs/one4000/NT/",1, network, "/home/callum/Desktop/ASF_Output/NT/4000/One/")

println("4000: 4x1000 Grid")
simulate_region("Inputs/four4000/NT/",4, network, "/home/callum/Desktop/ASF_Output/NT/4000/Grid_4/")

println("4000: 4x1000 Line")
network = path_graph(4)
simulate_region("Inputs/four4000/NT/",4, network, "/home/callum/Desktop/ASF_Output/NT/4000/Line_4/")

println("4000: 16x250 Grid")
network = grid([4,4])
simulate_region("Inputs/sixteen4000/NT/",4, network, "/home/callum/Desktop/ASF_Output/NT/4000/Grid_16/")

println("4000: 16x250 Line")
network = path_graph(16)
simulate_region("Inputs/sixteen4000/NT/",4, network, "/home/callum/Desktop/ASF_Output/NT/4000/Line_16/")