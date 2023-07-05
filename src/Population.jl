"Module to initialise population (spin up and seed ASF) for the three different models!"

module Population

using Distributions
using Random
using DifferentialEquations
using LinearAlgebra
using SparseArrays 
using QuadGK

export burn_population
export build_s

function spinup_and_seed(sim,pops, S0, Parameters, verbose)
    # wrapper function to burn in different populations to desired day, different methods for different models
    
    Mn = sim.Model #model number
    Np = Parameters.Populations.pop #number of populations!
    #Calculating the correct population at the start date

    if (Mn == 3) & (Np == 1) #Model 3 with single population
        S1 = burn_m3_single(sim,S0, Parameters) 
    elseif Mn == 3 #Model 3 with multiple population 
        S1 = burn_m3_full(sim,S0,Parameters)
    else #The homogeneous models
        S1 = burn_m1m2(sim,S0, Parameters)
    end
    
    #Seeding ASF in the population!
    U1 = seed_ASF(sim,pops, Parameters, S1, verbose)
    
    return U1
end


function seed_ASF(sim,pops, Parameters, S1, verbose)
    #function to Seed ASF in population!

    Mn = sim.Model
    K = Parameters.K
    n_classes = 5 #number of different classes

    if Mn != 3 #seedingin in single "group"
        K = K[1][1]
        data = pops[1]
        
        U1 = zeros(n_classes)

        p_e = data.N_e[1]/100 #percentage of K that are exposed!
        p_i = data.N_i[1]/100 #percentage of K that are infected!
        
        
        if (p_e + p_i) > 1
            if verbose
                @warn "Over 100% of Population Infected or Exposed! Setting to 1% each!" 
            end
                p_e = 0.01
                p_i = 0.01
        end
        
        ne = round(K*p_e)
        ni = round(K*p_i)
        
        U1[1] = S1[1] - ne - ni
        U1[2] = ne
        U1[3] = ni

    else #seeding in network!
        counts = Parameters.Populations
        
        n_pop = counts.pop #number of populations we have
        n_inf = counts.inf #population we will seed with asf!
        n_p = sim.N_Pop # number of populations per region

        if n_inf > n_pop 
            @warn "Wanting to seed in population $(n_inf), when there is only $(n_pop) populations. Seeding in 1"
            n_inf = 1 #will seed in first population instead!
        end 

        n_cs = counts.cum_sum #keeping track of groups in each population
        
        N_groups = n_cs[end] #total number of groups across all populations!
       
        U1 = Vector{Int16}(undef, N_groups*5) #vector to store all output population!

        for i in 1:n_pop #looping over populations
         
            Tg = n_cs[i+1]-n_cs[i] #number of groups in pop i
            u1 = zeros(Int32,5*Tg) #group pops for Population i
            u1[1:5:end] = S1[n_cs[i]+1:n_cs[i+1]] #assigning our S values
            
            
            if i == n_inf #in seeded population!
                
                network_inf = counts.networks[i] #network of infected population (used for accrute seeding)

                j = (i-1) ÷ n_p + 1
                data =  pops[j]

                p_e = data.N_e[1]/100 #percentage of K that are exposed!
                p_i = data.N_i[1]/100 #percentage of K that are infected!
                
                p_ei = p_e + p_i
                
                if p_ei > 1
                    if verbose
                        @warn "Over 100% of Population Infected or Exposed! Setting to 1% each!" 
                    end
                    p_e = 0.01
                    p_i = 0.01
                    p_ei = p_e + p_i 
                end
                
                n_groups = trunc(Int8,Tg*p_i)

                if verbose
                    @info "Seeding ASF in  $n_groups groups"
                end
                

                si = counts.cum_sum[i] #start index of pop
                ei = counts.cum_sum[i+1] #end index of pop

                seeded_groups = 0
               
                while seeded_groups == 0 
                    seeded_groups = find_nodes(network_inf,n_groups) #groups we are seeding ASF in
                end

                for j in seeded_groups
                    ki= K[i]
                    
                    g_p = ki[j] #group carrying capacity
                    
                    init_pop = S1[j + si]
                    ra = j -1 #needed for indexing!
                   
                    if g_p > 1 #sow population! (Asumming whole population infected or exposed!)
                        
                        e_pop = trunc(Int16, init_pop*p_e/p_ei)
                        i_pop = init_pop - e_pop

                        u1[ra*5+1] = 0 #S=0
                        u1[ra*5+2] = e_pop
                        u1[ra*5+3] = i_pop
                    else #boars!
                        e_prob = rand(1)[1] > p_e/p_ei #if it is E
                        if e_prob #seeded in e
                            u1[ra*5+1] = 1
                            u1[ra*5+3] = 0
                        else #seeded in i
                            u1[ra*5+1] = 0
                            u1[ra*5+3] = 1
                        end
                    end
                end
                
            end
            
            n_cs_class = n_cs*5
        
            U1[n_cs_class[i]+1:n_cs_class[i+1]] = u1
        end
            
    end

    return U1
end


function burn_m3_single(sim,S0, PP)

    ny = 5 #will allow for 5 years spinup time
    tspan = (0.0,ny*365+sim.S_day)

    NG = length(S0)
    eqs = 2 #number of processes

    #Matrix of all the transitions between classes for Gillespie model
    dc = sparse(zeros(NG,NG*eqs))
    dc[0*NG+1:eqs*NG+1:end] .=  1
    dc[1*NG+1:eqs*NG+1:end] .= -1

    function regular_c(du,u,p,t,counts,mark)  
        mul!(du,dc,counts)
        nothing
    end
    
    μ = mean(PP.μ_p[1][end]) #taking value from last year!
    k = birthpulse_norm(PP.bw[1], μ)
    params = [μ, PP.K[1], 0, PP.g[1], PP.bw[1], PP.bo[1], k, PP.Populations.networks[1]]
    
    rj_burn = RegularJump(asf_model_burn_multi, regular_c, eqs*NG)
    prob_burn = DiscreteProblem(S0,tspan,params)
    jump_prob_burn = JumpProblem(prob_burn, Direct(), rj_burn)    
    sol_burn = solve(jump_prob_burn, SimpleTauLeaping(),dt=1);

    S1 = reduce(vcat,transpose.(sol_burn.u))[end,:]
   
    return S1
end

function burn_m3_full(sim,S0, PP)

    ny = 5 #will allow for 5 years spinup time
    tspan = (0.0,ny*365+sim.S_day)

    NG = length(S0)
    eqs = 2 #number of processes

    #Matrix of all the transitions between classes for Gillespie model
    dc = sparse(zeros(NG,NG*eqs))
    dc[0*NG+1:eqs*NG+1:end] .=  1
    dc[1*NG+1:eqs*NG+1:end] .= -1

    function regular_c(du,u,p,t,counts,mark)  
        mul!(du,dc,counts)
        nothing
    end
  
    rj_burn = RegularJump(asf_model_burn_multi_full, regular_c, eqs*NG)
    prob_burn = DiscreteProblem(S0,tspan,PP)
    jump_prob_burn = JumpProblem(prob_burn, Direct(), rj_burn)    
    sol_burn = solve(jump_prob_burn, SimpleTauLeaping(),dt=1);

    S1 = reduce(vcat,transpose.(sol_burn.u))[end,:]
   
    return S1
end

function burn_m1m2(sim,U0, PP)
    #Function to run homogeneous burn in!
    ny = 10 #will allow for 10 years spinup time
    tspan = (0.0,ny*365+sim.S_day)
#taking value from last year!
    params = [PP.μ_p[1][end][1],PP.K[1][1],0,PP.bw[1],PP.bo[1], PP.k[1][end]] #\simga = 0 allows for faster return to correct K!
     #only need to run on S, so can not use others!
    prob_ode = ODEProblem(asf_model_burn_single, U0, tspan,params)
    sol = solve(prob_ode, saveat = ny*365+sim.S_day,reltol=1e-8)
    
    return trunc.(Int16,sol.u[2]) #our final pop!
end

function asf_model_burn_single(du,u,p,t)
    #ode equivelent of our ASF model
    μ_p, K, σ, bw, bo, k  = p 
    S = u[1]
    ds = μ_p*(σ + ((1-σ))*sqrt(S/K))
       
    du[1] = k*exp(-bw*cos(pi*(t+bo)/365)^2)*(σ * S + ((1-σ)) * sqrt(S * K)) - ds*S
    
    nothing
end

function asf_model_burn_multi(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    μ_p, K, σ, g, bw, bo, k, pop_con  = p 
    
    u[u.<0] .= 0
    
    tg = length(u)
    
    p_mag = birth_pulse_vector(t,k,bw,bo)
   
    Births =  @. p_mag * (σ * u + ((1-σ)) * sqrt(u) * sqrt(K))#total! (rate times NP)
   
    #now stopping and transferring boar births
    mask_boar = (K .== 1) .& (u .> 0) #boars with a positive population
    boar_births = p_mag*sum(mask_boar)
    Births[mask_boar] .= 0
    mask_p_s = (u .> 1) .& (K .> 1) #moving it to postive 
    Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
    
    #now allowing for inter group migration births
    n_empty  = sum(u .== 0 ) 
    n_r = (n_empty/tg)^2

    if (n_empty/tg) > 0.01
        
        dd = copy(u)
        dd[dd .< 2] .= 0  #Groups with 3 or more pigs can have emigration
        connected_pops = pop_con * dd 
        mask_em =  (dd .> 0) #populations that will have emigration

        em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

        mask_im = (u .== 0) .& (connected_pops .> 1) #population zero but connected groups have 5 or more pigs

        Births[mask_em] .*= n_r
        Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)

    end 
   
    out[1:2:end] .= Births
    out[2:2:end] .= u.* μ_p.*(σ .+ (1-σ).*sqrt.(u./K))*g
   
    nothing
end

function asf_model_burn_multi_full(out,u,p,t)

    year = 365 #days in a year

    u[u.<0] .= 0
   
    pop_data = p.Populations
 
    for i in 1:pop_data.pop #looping through populations!
        
        si = pop_data.cum_sum[i]+1 #start index of pop
        ei = pop_data.cum_sum[i+1] #end index of pop

        S = Vector{UInt8}(u[si:ei]) #population we want!

        K = p.K[i]
        
        tg = length(S) #total groups in all populations
        
        p_mag = @. p.k[i][end]*exp(-p.bw[i]*cos(pi*(t+p.bo[i])/365)^2) #birth pulse value at time t
        Births = @. p_mag*(0.75 * S + ((0.25)) * sqrt(S) * sqrt(K))#total! (rate times NP)

        #now stopping boar births
        mask_boar = (K .== 1) .& (S .> 0) #boars with a positive population
        boar_births = p_mag*sum(mask_boar)
        Births[mask_boar] .= 0
        mask_p_s = (S .> 1) .& (K .> 1) #moving it to postive sow groups with at least 2 pigs
        Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
        
        n_empty  = sum(S .== 0 )   
        
        if n_empty/tg > 0.01   #migration births (filling dead nodes if there is a connecting group with 2 or more pigs)
            
        connections = shuffle(pop_data.inter_connections[i])

        for i in eachrow(connections)
            g1 = i[1]
            g2 = i[2]

            if (S[g1] > 2) & (S[g2] == 0) & (Births[g1] > p_mag) #g1 to g2 
                Births[g1] -= p_mag
                Births[g2] += p_mag
            elseif (S[g2] > 2) & (S[g1] == 0) & (Births[g2] > p_mag) #g2 to g1
                Births[g2] -= p_mag
                Births[g1] += p_mag
            end
        end     

        end 
      
        out[2*(si-1)+1:2:2*ei] .= Births
        out[2*(si-1)+2:2:2*ei] .=  @. S *  p.μ_p[i][end]*(0.75 + ((0.25))*sqrt(S/K))*p.g[i] #rate
       
    end

    nothing
end

function birth_pulse_vector(t,k,s,p)
    #birth pulse not for a vector
    return k*exp(-s*cos(pi*(t+p)/365)^2)
end

function build_s(sim, pops, counts, verbose)
    #=
    Function to build the initial S population (seed at a later date) 
    =#
    
    n_p = sim.N_Pop # number of populations per region
    n_r = size(pops)[1] #number of regions
    n_pops = n_p*n_r #total number of populations

    n_cs = counts.cum_sum
    
    MN = sim.Model #which model we will be running with
    N_groups = n_cs[end]
   
    s_total = Vector{Int16}(undef, N_groups) #vector to store initial populationss
    densities = Vector{Float32}(undef, n_pops) #vector to store each population's density
    areas = Vector{Float32}(undef,n_pops) #vector to store each population's area
    
    max_density = 100
    max_group_size = 1000000
    min_group_size = 2
    
    if MN == 3
        
        for j in 1:n_pops #looping through all populations
            network = counts.networks[j]
            i = (j-1) ÷ n_p + 1

            #population data
            data = pops[i]
            boar = data.Boar_p[1] #percentage of groups that are solitary boar
            
            #Density of population
            Density_D = TruncatedNormal(data.Dense[1],data.Dense[2],0,max_density)
            Density = rand(Density_D)
            densities[j] = Density
            
            #vector to store  population numbers for each class of each group
            s_pop = zeros(Int32,(n_cs[i+1]-n_cs[i]))
            
            N_feral = counts.feral[i] #number of feral groups 
            N_boar = round.(Int16, N_feral .* boar)#number of wild boar in pop
            N_sow = N_feral - N_boar  #number of sow groups in pop
            N_farm = counts.farm[i] #number of farms in population

            sow_dist = TruncatedNormal(data.N_f[1],data.N_f[2],min_group_size, max_group_size) #dist for number of pigs in sow feral group
            sow_groups = round.(Int16,rand(sow_dist, N_sow)) #drawing the populations of each feral group in population
        
            pop_network = copy(network[1:N_feral,1:N_feral]) #isolating network for this feral pop 


            pop_network[pop_network .!= 0] .= 1 #seeting all 
            group_degree = vec(sum(Int16, pop_network, dims = 2)) .- 1 #group degree -1 as not counting inta group connections 

            #now want to choose the N_boar groups with the highest degree as these are boars, the others will be sow_dist
            index_boar = sort(partialsortperm(group_degree,1:N_boar,rev=true)) #index of all boar groups
            index_sow = setdiff(1:N_feral, index_boar) #index of all sow groups
            

            s_pop[index_sow] = sow_groups
            s_pop[index_boar] .= 1
        
            
            total_pop = sum(s_pop)
            total_area = total_pop/Density
            
            areas[j] = total_area
            
            #livestock populations
            
            if N_farm > 0
                group_farm_pops = TruncatedNormal(data.N_l[1],data.N_l[2],1,10000)#distribution of group sizes         
                farm_groups = round.(Int,rand(group_farm_pops, N_farm))
                s_pop[N_feral:end] = farm_groups
            end
            
            
            s_total[n_cs[j]+1:n_cs[j+1]] = s_pop
            
        end
    else
        
        data = pops[1]
        n_g = data.N_feral[1] #number of feral groups (assuming M3)
        b_p = data.Boar_p[1] #percentage of groups that are solitary boar\
        s_g = data.N_f[1] #sow group size
       
        homogeneous_pop = n_g*b_p + s_g*n_g*(1-b_p)
        
        s_total = [homogeneous_pop]

        Density_D = TruncatedNormal(data.Dense[1],data.Dense[2],0,max_density)
        Density = rand(Density_D)
        densities[1] = Density
        
        total_area = homogeneous_pop/Density
        areas[1] = total_area

    end

    return trunc.(Int,s_total), densities, areas
    
end


function find_nodes(Network, N_connections)
  
    p1g = zeros(Int16,0) #groups we are using
    p1c = zeros(Int16,0) #centre groups that we search for links from (subet of p1g)

    g_p = rand(1:size(Network)[1]) #base group of p1 that is doing the connecting! now need to find n neighbours


    append!(p1g, g_p)
    append!(p1c, g_p)

    if N_connections > 1 #we need more connections!

        netw = Network[:,g_p] #connected populations to randomly selected group
        cons = findall(>(0), netw) #all connectedions of randomly selected group

        if N_connections -1 < length(cons) #the neighbours more neighbours than required
            append!(p1g,shuffle(cons)[1:(N_connections- 1)])
        elseif N_connections -1 == length(cons) #exactly the right amount of neighbours
            append!(p1g,cons)
        else #we need more neighbours... need to pick one of the connections that is not the primary
            append!(p1g,cons) #first append all neighbours

            while N_connections > length(p1g)
                non_central_groups = setdiff(p1g,p1c)
                if isempty(non_central_groups)
                    @warn "No Groups left to choose, will try again"
                    return 0
                end
                new_central = rand(non_central_groups)

                append!(p1c,new_central)

                netw = Network[:,new_central] #connected populations to randomly selected group
                cons = findall(>(0), netw) #all connectedions of randomly selected group

                unique_cons = setdiff(cons, p1g)

                t_cons = length(unique_cons) + length(p1g)
                if N_connections <= t_cons
                append!(p1g,shuffle(unique_cons)[1:N_connections-length(p1g)])
                else 
                    append!(p1g,unique_cons)
                end
            end

        end
    end
    
    return p1g

end
function birthpulse_norm(s, DT)
   
    integral, err = quadgk(x -> exp(-s*cos(pi*x/365)^2), 1, 365, rtol=1e-8);
    k = (365*DT)/integral 
    
    return k
end

end
