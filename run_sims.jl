using ASF
using Distributions
using DelimitedFiles
using CSV

function simulate_births_deaths(ip)

    save = "/test"
    if !isdir(ip*save)
        mkdir(ip*save)
    end

    n_sims = 10

    births = rand(Uniform(0.5,0.95),n_sims)
    decay = rand(Uniform(0.5,1.5),n_sims)

    writedlm(ip*save*"/births.csv", births, ',')
    writedlm(ip*save*"/decay.csv", decay, ',')

    #output = zeros(Int16,n_sims) 

    for i in 1:n_sims
        bm = births[i]
        dm = decay[i]
	    out = Model_sim(ip, adj = [bm, dm])
        sim_data = out.u
        
        if i == 1
            writedlm(ip*save*"/Sim_output.csv", [sum(sim_data .> 0)], ',')
        else
            CSV.write(ip*save*"/Sim_output.csv", (data = [sum(sim_data .> 0)],), append = true)
        end
    end
end

#println("ACT")
#simulate_births_deaths("Inputs/ACT/")
println("NSW")
simulate_births_deaths("Inputs/NSW/")
println("QLD")
simulate_births_deaths("Inputs/QLD/")
println("NT")
simulate_births_deaths("Inputs/NT/")

