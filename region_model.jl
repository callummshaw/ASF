using ASF
using JLD2
using DifferentialEquations

function simulate_region(ip)

    save = "/onekone/"
    if !isdir(ip*save)
        mkdir(ip*save)
    end

    μfy = [0.5,0.6,0.7,0.8,0.9]

    for i in μfy
        println(i)
        sn = Int(i*10)
        sim_output = Model_sim(ip, adj = μfy[1])
        save_object(ip*save*"out$(sn).jdl2",sim_output.u)
    end
end


println("ACT")
simulate_region("Inputs/ACT/")
println("NSW")
simulate_region("Inputs/NSW/")
println("QLD")
simulate_region("Inputs/QLD/")
println("NT")
simulate_region("Inputs/NT/")