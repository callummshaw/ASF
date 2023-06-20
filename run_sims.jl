using ASF
using Distributions
using DelimitedFiles
using CSV
using DataFrames

function simulate_births_deaths(ip)

    save = "/test2"
    if !isdir(ip*save)
        mkdir(ip*save)
    end

    n_sims = 500
    input_data = zeros(Float64,n_sims,2)
    input_data[:,1] = rand(Uniform(0.5,0.95),n_sims)
    input_data[:,2] = rand(Uniform(0.5,1.5),n_sims)
    CSV.write(ip*save*"/input_arrays.csv", DataFrame(input_data, :auto), header=["First Year Mortality", "Decay Multiplier"])

    for i in 1:n_sims
       
	    out = Model_sim(ip, adj = input_data[i,:])
        sim_data = out.u
        
        if i == 1
            writedlm(ip*save*"/Sim_output.csv", [sum(sim_data .> 0)], ',')
        else
           CSV.write(ip*save*"/Sim_output.csv", (data = [sum(sim_data .> 0)],), append = true)
        end
    end
end

println("ACT")
simulate_births_deaths("Inputs/ACT/")
println("NSW")
simulate_births_deaths("Inputs/NSW/")
println("QLD")
simulate_births_deaths("Inputs/QLD/")
println("NT")
simulate_births_deaths("Inputs/NT/")