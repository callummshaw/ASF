module ASF_Init

using DifferentialEquations
using LinearAlgebra
using SparseArrays 

export burn_population
#module for burning in of population

function burn_population(sim,pops, S0,Parameters)
    # wrapper function to burn in different populations to desired day
    Mn = sim.Model

    if Mn == 3
        S1 = burn_m3(sim,S0, Parameters)
    else
        S1 = burn_m1m2(sim,S0, Parameters)
    end
    
    U1 = seed_ASF(sim,pops, Parameters, S1)

    return U1
end


function seed_ASF(sim,pops, Parameters, S1)
    #function to Seed ASF in population!

    Mn = sim.Model
    K = Parameters.K
    n_classes = 5 #number of different classes

    if Mn != 3 #seedingin in single "group"

        data = pops[1]
        
        U1 = zeros(n_classes)

        p_e = data.Npe #percentage of K that are exposed!
        p_i = data.Npe #percentage of K that are infected!
        
        if p_e _+ p_i > 1
            @warn "Over 100% of Population Infected or Exposed! Setting to 1% each!" 
            p_e = 0.01
            p_i = 0.01
        end

        ne = K*p_e
        ni = K*p_i

        U1[1] = S1[1] - ne - ni
        U1[2] = ne
        U1[3] = ni

    else #seeding in network!
        counts = Parameters.Population
        
        n_pop = counts.pop #number of populations
        n_inf = counts.inf #population we will seed with asf!

        if n_inf > n_pop #wanting to seed in a population that does not exist! This could be simply expanded to include multiple seeding populations!
            n_inf = 1 #will seed in first population instead!
        end 

        #=
        if n_inf > n_pop
            n_inf = n_pop
        end
        n_in = shuffle(1:n_pop)
        n_d = n_in[1:n_inf] #array of random populations to seed ASF!
        =#

        n_cs = counts.cum_sum #keeping track of groups in each population
        
        N_groups = n_cs[end] #total number of groups across all populations!
       
        
        U1 = Vector{Int16}(undef, N_groups*5) #vector to store all output population!

        for (i, data) in enumerate(pops)
            
            p_e = data.Npe #percentage of groups that are exposed!
            p_i = data.Npe #percentage of groups that are infected!
            
            p_ei = p_e + p_i

            if p_ei > 1
                @warn "Over 100% of Population Infected or Exposed! Setting to 1% each!" 
                p_e = 0.01
                p_i = 0.01
                p_ei = p_e + p_i 
            end
            
            Tg = n_cs[i+1]-n_cs[i]
            u1 = zeros(Int32,N_class*Tg) #group pops for Population i
            u1[1:5:end] = S1[n_cs[i]+1:n_cs[i+1]] #assigning our S values
            
            if i == n_inf #in seeded population!
                
                n_seed = trunc(Int16,Tg*(p_ei))  #number of groups we will seed ASF in!

                rr = rand(n_cs[i]+1:n_cs[i+1]) #selecting 1 random group in population
                
                netw = Parameters.β_d[:,rr] #connected populations to randomly selected group
                cons = findall(>(0), netw)
                
                asf_groups = Vector{Int16}
                append!(asf_groups,rr)
                
                n_seed = length(n_cons) #number of connections(populations we will seed in)
                
                if n_cons == (n_seed -1) #there are exactly the right ammount of connections!
                
                    append!(asf_groups,cons)

                elseif n_cons > (n_seed - 1)  #there are more connected groups than we want to seed in!
                    
                    append!(asf_groups,shuffle(cons)[1:(n_seed- 1)])

                    
                else #annoyingly there we want to seed ASF in more groups than we have connections from initial seeded group
                    
                    append!(asf_groups,cons) # we want all groups! but will need more...

                    while length(asf_groups) < n_seed  #will seed in another population!
                        
                        rx = rand(n_cs[i]+1:n_cs[i+1]) #selecting 1 random group in population
                        
                        while rx ∉ asf_groups #making sure group hasnt already been selected 
                            rx = rand(n_cs[i]+1:n_cs[i+1]) 
                        end
                        
                        append!(asf_groups,rx) #store this group

                        if length(asf_groups) == n_seed #checking this last group doesnt put us over the limit!
                            break
                        end

                        netw = Parameters.β_d[:,rx] #connected populations to randomly selected group
                        cons = setdiff(findall(>(0), netw),asf_groups) # new connections!
                        
                        n_cons = length(cons)

                        if n_cons + length(asf_groups) > n_seed #need slightly less 
                            append!(asf_groups, shuffle(cons)[1:(n_seed-length(asf_groups))])
                        else #we have exact amount or another trip around the loop!
                            append!(asf_groups,cons)    
                        end
                    end 
                    
                end 
                
                for i in asf_groups #now seeding in all the groups we want!
                    g_p = K[i] #group carrying capacity
                    
                    ra = i -1 #needed for indexing!
                    
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


            else #not in seeded population!
                u1[1:5:end] = S1[n_cs[i]+1:n_cs[i+1]] #simply re-assign our S values into our U vector!
            end


            n_cs_class = n_cs*5
        
            U1[n_cs_class[i]+1:n_cs_class[i+1]] = u1
        end
            
        
    end

    return U1
end


function burn_m3(sim,S0, PP)

    ny = 5 #will allow for 10 years spinup time
    tspan = (0.0,ny*365+sim.S_day)

    NG = length(S0)
    eqs = 2 #number of processes

    #Matrix of all the transitions between classes for Gillespie model
    dc = sparse(zeros(NG,NG*eqs))
    dc[0*NG+1:eqs*NG+1:end] .=  1
    dc[1*NG+1:eqs*NG+1:end] .= -1

    params = [PP.β_b[1], PP.μ_p[1], PP.K[1], 0, PP.θ[1], PP.g[1], PP.bw[1], PP.bo[1], PP.k[1]]
    
    rj_burn = RegularJump(asf_model_burn_multi, regular_c, eqs*NG)
    prob_burn = DiscreteProblem(S0,tspan,params)
    jump_prob_burn = JumpProblem(prob_burn, Direct(), rj_burn)
    sol_burn = solve(jump_prob_burn, SimpleTauLeaping(),dt=1, saveat = ny*365+sim.S_day);

    S1 = reduce(vcat,transpose.(sol_burn.u))[end,:]

    return S1
end


function regular_c(du,u,p,t,counts,mark)  
    mul!(du,dc,counts)
    nothing
end


function burn_m1m2(sim,U0, PP)
    #Function to run homogeneous burn in!
    ny = 10 #will allow for 10 years spinup time
    tspan = (0.0,ny*365+sim.S_day)
    params = [PP.μ_p[1],PP.K[1],0,PP.bw[1],PP.bo[1], PP.k[1]] #\simga = 0 allows for faster return to correct K!
    Y0 = U0[1] #only need to run on S, so can not use others!
    prob_ode = ODEProblem(asf_model_burn_single, Y0, tspan,params)
    sol = solve(prob_ode, saveat = ny*365+sim.S_day,reltol=1e-8)
    
    return sol.u[2] #our final pop!
end

function asf_model_burn_single(du,u,p,t)
    #ode equivelent of our ASF model
    μ_p, K, σ, bw, bo, k  = p 
    
    S = u

    ds = μ_p*(σ + ((1-σ))*sqrt(S/K))
        
    du[1] = k*exp(-bw*cos(pi*(t+bo)/365)^2)*(σ .* S .+ ((1-σ)) .* sqrt.(S .* K)) - ds*S
    
    nothing
end

function asf_model_burn_multi(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β_b, μ_p, K, σ, θ, g, bw, bo, k  = p 
    
    
    u[u.<0] .= 0
    tg = length(u)
    
    p_mag = birth_pulse_vector(t,k,bw,bo)

    if θ == 0.5 #density births and death!
        Deaths = μ_p.*(σ .+ ((1-σ)).*sqrt.(abs.(Np./K)))*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* sqrt.(abs.(Np .* K)))#total! (rate times NP)
    else
        Deaths = μ_p.*(σ .+ ((1-σ)).*(Np./K).^θ)*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* Np.^(1-θ) .* K.^θ)#total! (rate times NP)
    end

    #now stopping boar births
    mask_boar = (K .== 1) .& (u .> 0) #boars with a positive population
    boar_births = p_mag*sum(mask_boar)
    Births[mask_boar] .= 0
    mask_p_s = (u .> 1) .& (K .> 1) #moving it to postive 
    Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
    
    n_empty  = sum(u .== 0 ) 
    n_r = (n_empty/tg)^2
    
    
    if (n_empty/tg) > 0.01
        
        dd = copy(u)
        dd[dd .< 2] .= 0  #Groups with 3 or more pigs can have emigration
        connected_pops = β_b * dd 
        mask_em =  (dd .> 0) #populations that will have emigration

        em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

        mask_im = (u .== 0) .& (connected_pops .> 1) #population zero but connected groups have 5 or more pigs

        Births[mask_em] .*= n_r
        Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)

    end 
    out[1:2:end] .= Births
    out[2:2:end] .= u.*Deaths
   
    nothing
end

function birth_pulse_vector(t,k,s,p)
    #birth pulse not for a vector
    return k*exp(-s*cos(pi*(t+p)/365)^2)
end

end