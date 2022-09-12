module ASF_Inputs

using LinearAlgebra
using DelimitedFiles
using DataFrames
using CSV
using FileIO
using LinearAlgebra
using NPZ
using Random, Distributions
using Graphs

export Model_Data

abstract type Data_Input end

struct Meta_Data <: Data_Input
    #=
    Structure to store all the meta data infomation for the run:
    number of years, if the model is being run in ensemble, if
    using distros or just mean values for parameters, the number of 
    populations and the number of said populations seeded with ASF
    =#
    years::Float16 #years simulation will run for
    N_ensemble::Int16 #number of runs in an ensemble
    Identical::Bool #if we params to be drawn from dist or just means of dist
    N_Pop::Int8 #number of populations, must match the number of population files in input
    N_Inf::Vector{Int8} #number of populations init with ASF
    N_Seed::Int8
    C_Type::String #what kind of connection we want to init with between populations, line (l), circular (c), total (t), or off (o)
    C_Str::Float32 #strength of the connections between populations, note this is only for l,c,t
    Network::String #Type of network used for model random (r), scale-free (s), or small worlds (w)

    function Meta_Data(input, numv)
        Ny = parse(Int16, input.Value[1])
        Ne = parse(Int16,input.Value[2])
        I = (input.Value[3] == "true")
        Np = parse(Int8, input.Value[4])
        Ni = numv
        Ns = parse(Int16, input.Value[6])
        Ct = input.Value[7]
        Cs = parse(Float32, input.Value[8])
        Nw = input.Value[9]
        new(Ny,Ne,I,Np,Ni,Ns,Ct,Cs,Nw)
    end
end

struct Population_Data <: Data_Input
    
    Dense::Vector{Float64} #density of population
    N_feral::Vector{Int16} #number of feral groups
    N_farm::Vector{Int8} #number of farm groups
    N_int::Vector{Int8} #average interconnection between feral groups
    B_f::Vector{Float64} #intra feral group transmission
    B_l::Vector{Float64} #intra farm transmission
    B_ff::Vector{Float64} #inter feral group transmission
    B_fl::Vector{Float64} #farm-feral transmission
    Death::Vector{Float64} #ASF death prob
    Recovery::Vector{Float64}#Recovery rate
    Latent::Vector{Float64} #latent period
    Corpse::Vector{Float64} #corpse infection modifier
    Decay_l::Vector{Float64} #decay farm
    Decay_f::Vector{Float64} #decay feral
    N_f::Vector{Float64} #feral population size
    N_l::Vector{Float64} #farm population size
    N_e::Vector{Float64} #number of exposed in seeded
    N_i::Vector{Float64} #number of infected in seeded
    Birth::Vector{Float64} #birth rate
    Death_n::Vector{Float64} #natural death rate
    Immunity::Vector{Float64}

    function Population_Data(input)
        
        Den = [input.Mean[1],input.STD[1]]
        Nf = [input.Mean[2],input.STD[2]]
        Nl = [input.Mean[3],input.STD[3]]
        Ni = [input.Mean[4],input.STD[4]]
        Bf = [input.Mean[5],input.STD[5]]
        Bl = [input.Mean[6],input.STD[6]]
        Bff = [input.Mean[7],input.STD[7]]
        Bfl = [input.Mean[8],input.STD[8]]
        D = [input.Mean[9],input.STD[9]]
        R = day_to_rate(input.Mean[10],input.STD[10])
        L = day_to_rate(input.Mean[11],input.STD[11])
        C = [input.Mean[12],input.STD[12]]
        Dl = day_to_rate(input.Mean[13],input.STD[13])
        Df = day_to_rate(input.Mean[14],input.STD[14])
        Npf = [input.Mean[15],input.STD[15]]
        Npl = [input.Mean[16],input.STD[16]]
        Npe = [input.Mean[17],input.STD[17]]
        Npi = [input.Mean[18],input.STD[18]]
        B = [input.Mean[19],input.STD[19]]
        Dn = [input.Mean[20],input.STD[20]]
        Im = day_to_rate(input.Mean[21],input.STD[21])
        
        new(Den, Nf, Nl, Ni ,Bf, Bl, Bff, Bfl, D, R, L, C, Dl, Df, Npf, Npl, Npe, Npi, B, Dn, Im)
        
    end
    
end

mutable struct Network_Data
    #=
    Structure to store key data on each population, internal just to keep track of a few params
    =#
    feral::Vector{Int16}
    farm::Vector{Int8}
    pop::Int8
    total::Vector{Int16}
    cum_sum::Vector{Int16}
    
    inf::Vector{Int8}
    
    density::Vector{Float32}
    area::Vector{Float32}

    function Network_Data(feral,farm, inf, density,area)
        n = size(feral)[1]
        t = feral + farm
        cs = pushfirst!(cumsum(t),0)
        new(feral,farm,n,t,cs, inf, density, area)
    end
end 


mutable struct Model_Parameters
    #=
    Structure to store key parameters
    =#
    
    β::Matrix{Float32} #transmission matrix
    β_b::Matrix{Int16} #what feral groups are linked to each other, for births
    β_d::Matrix{Int16} #used to identify feral groups and farms for each population for density calcualtions
    
    μ_b::Vector{Float32} #birth rate
    μ_d::Vector{Float32} #natural death rate
    μ_c::Vector{Int16} #carrying capicity
    
    ζ::Vector{Float32} #latent rate
    γ::Vector{Float32} #recovery rate
    ω::Vector{Float32} #corpse infection modifier
    ρ::Vector{Float32} #death probability
    λ::Vector{Float32} #corpse decay rate
    κ::Vector{Float32} #loss of immunity rate


    Populations::Network_Data #breakdown of population
    
    function Model_Parameters(sim, pops, U0, Populations, network)
        
        β, connected_pops, connected_births = beta_construction(sim, pops, Populations, network)
        μ_birth, μ_death, μ_capicty, ζ, γ, ω, ρ, λ, κ = parameter_build(sim, pops, U0, Populations)
        
        new(β, connected_births, connected_pops, μ_birth, μ_death, μ_capicty, ζ, γ, ω, ρ, λ, κ, Populations)
    end
    
end

struct Model_Data
    #=
    Structure to store key data on mode
    =#
    Time::Tuple{Float32, Float32} #Model run time
    U0::Vector{Int32} #Initial Population
    Parameters::Model_Parameters #Model parameters
    Populations_data::Vector{Population_Data} #distributions for params

    function Model_Data(Path)
        
        sim, pops = read_inputs(Path)
     
        Time = (0.0,sim.years[1]*365)
       
        #now building feral pig network
        network, counts = build_network(sim, pops) 
        
        #Now using network to build init pops
        U0 = build_populations(sim, pops, network, counts) #initial populations
     
        Parameters = Model_Parameters(sim, pops, U0, counts, network)
        
        new(Time, U0, Parameters, pops)
        
    end
    
end

function build_network(sim, pops)
    #This function builds the network
    n_pops = sim.N_Pop #number of populations
    network = Vector{Matrix{Int16}}(undef, n_pops) #vector to store all the 
    feral_pops = Vector{Int16}(undef, n_pops)
    farm_pops = Vector{Int16}(undef, n_pops)

    for pop in 1:n_pops 
        
        data = pops[pop]

        #Feral dist
        if data.N_feral[1] == 0
            nf = 0
        else
            nf_d = TruncatedNormal(data.N_feral[1],data.N_feral[2],0,1000) #number of feral group distribution
            nf = trunc(Int16,rand(nf_d))
        end
       
        #Farm dist     
        if data.N_farm[1] == 0
            nl = 0
        else
            nl_d = TruncatedNormal(data.N_farm[1],data.N_farm[2],0,100) #number of farms distribution
            nl =  trunc(Int16,rand(nl_d))
        end
        
        feral_pops[pop] = nf
        farm_pops[pop] = nl

        n_o = nf -1 #number of other feral groups; excluding "target group"

        if data.N_int[1] > n_o
           # @warn "Warning interconnectedness exceeds number of feral groups in population; average enterconnectedness set to total  number of feral groups - 1"
            n_aim = n_o 
        else
            n_aim = data.N_int[1]
        end

    
        #this is where we buidl the three types of network, default is random but can allow for other network types

        if sim.Network == "s" #only works with even degrees 
            
            println(" Barabasi Albert Scale Free Network")
            if isodd(n_aim)
                @warn "Odd group degree detected, Scale free and small worlds require even degree"
            end
            
            adds = n_aim ÷ 2
            feral = barabasi_albert(nf, adds)

        elseif sim.Network == "w" #only works with even degrees
            
            println("Watts Strogatz Small Worlds Network")
            if isodd(n_aim)
                @warn "Odd group degree detected, Scale free and small worlds require even degree"
            end

            level = 0.5
            feral = watts_strogatz(nf, n_aim, level)
            
        else
            println("Erdos Renyi Random Network")
            #using an erdos-renyi random network to determine inter-group interactions
            p_c = n_aim / n_o #probability of a connection

            if p_c == 1
                feral = erdos_renyi(nf,sum(1:n_o)) #every group connected with every other
            else
                feral = erdos_renyi(nf,p_c) #not all groups are connected to every other
            end
        end
        

        network_feral = Matrix(adjacency_matrix(feral))*200 #inter feral = 2
        network_feral[diagind(network_feral)] .= 100 #intra feral = 1
        if nl != 0  #there are farm populations 
            
            N = nf + nl
            network_combined = zeros((N,N))
            network_combined[1:nf,1:nf] = network_feral 
           
            #currently assumes 1 farm-group interaction term

            for i in nf+1:N
                feral_pop = rand(1:nf) # the feral population the farm can interact within
                network_combined[i,i] = 400 #transmission within farm pop = 4

                network_combined[i,feral_pop] = 300 #feral-farm = 3
                network_combined[feral_pop, i] = 300 #feral-farm = 3

            end
        
            network[pop] = network_combined

        else #no farms in this population
            network[pop] = network_feral
        end
        
    end
    counts = Network_Data(feral_pops,farm_pops, sim.N_Inf,[0.0],[0.0])
    if n_pops > 1
        combined_network = combine_networks(network,sim,counts)
    else
        combined_network = network[1]
    end
    
    return combined_network, counts

end

function combine_networks(network,sim, counts)
    #we have generated all the networks, but they need to be combined!
    connections = population_connection(counts,sim) #calls to find connections

    n_pops = counts.pop
    n_cs = counts.cum_sum
    N = n_cs[end]

    meta_network = zeros(N,N)
  
    links = []
    
    for i in 1:n_pops #looping through populations

        meta_network[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]] = network[i]
        
        for j in connections[i]
            
            new_link = sort([i,j]) #checking if we have already linked the two populations

            if new_link ∉ links
                #println(new_link)
                push!(links, sort([i,j])) #storing the link
                
                if sim.C_Type == "o" #custom strengths
                    println("-------------------------------------")
                    println("Strength of Population  $(i) to Population  $(j) Transmission:")
                    str = readline()
                    str = parse(Float32, str)
                else #pre-determined strength
                    str = sim.C_Str
                end
                #the chosen population
                ll = n_cs[i] + 1
                ul = n_cs[i] + counts.feral[i]
                l_group = rand(ll:ul)

                #populations the chosen population is linked too
                ll_s = n_cs[j] + 1
                ul_s = n_cs[j] + counts.feral[j]

                s_group = rand(ll_s:ul_s)

                meta_network[l_group,s_group] = str
                meta_network[s_group,l_group] = str
                
            end

        end
    end

    return meta_network

end

function day_to_rate(Mean, STD)
    mean_rate = 1/Mean
    std_rate = 1/Mean - 1/(Mean+STD)

    return [mean_rate, std_rate]
end

function parameter_build(sim, pops, init_pops, counts)
    #=
    Function that builds most parameters for model
    =#
    birth_death_mod = 0.8
   
    K = init_pops[1:5:end] + init_pops[2:5:end] + init_pops[3:5:end] #carrying capacity of each group
    
    # All other params
    n_groups = length(K)
    ζ = Vector{Float32}(undef, n_groups) #latent rate
    γ = Vector{Float32}(undef, n_groups) #recovery/death rate
    μ_b = Vector{Float32}(undef, n_groups) #births
    μ_d = Vector{Float32}(undef, n_groups) #natural death rate
    μ_c = Vector{Int32}(undef, n_groups) #density dependent deaths
    ω = Vector{Float32}(undef, n_groups) #corpse infection modifier
    ρ = Vector{Float32}(undef, n_groups) #ASF mortality
    λ = Vector{Float32}(undef, n_groups) #corpse decay rate
    κ = Vector{Float32}(undef, n_groups)

    for i in 1:counts.pop
        data =  pops[i]
        
        nf = counts.feral[i]
        nl = counts.farm[i]
        nt = counts.total[i]
        
        cs = counts.cum_sum
        
        if sim.Identical == true #if running off means
            
            ζ[cs[i]+1:cs[i+1]] .= data.Latent[1]
            γ[cs[i]+1:cs[i+1]] .= data.Recovery[1]
            μ_b[cs[i]+1:cs[i+1]] .= data.Births[1]
            μ_d[cs[i]+1:cs[i+1]] .= birth_death_mod*data.Births[1]
            μ_c[cs[i]+1:cs[i+1]] = K[cs[i]+1:cs[i+1]]
            ω[cs[i]+1:cs[i+1]] .= data.Corpse[1]
            ρ[cs[i]+1:cs[i+1]] .= data.Death[1]
            κ[cs[i]+1:cs[i+1]] .= data.Immunity[1]
            λ[cs[i]+1:cs[i]+nf] .= data.Decay_f[1]
            λ[cs[i]+nf+1:cs[i+1]] .= data.Decay_l[1]

        else #running of distros
            
            ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2], 0, 5) #latent dist
            γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2], 0, 5) #r/d rate dist
            μ_b_d = TruncatedNormal(data.Birth[1], data.Birth[2], 0, 1) #birth dist
            #μ_d_d = TruncatedNormal(data.Death_n[1], data.Death_n[2], 0, 1) #n death dist
            ω_d = TruncatedNormal(data.Corpse[1], data.Corpse[2], 0, 1) #corpse inf dist
            ρ_d = TruncatedNormal(data.Death[1], data.Death[2], 0, 1) #mortality dist
            λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2], 0, 1) #corpse decay feral dist
            λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2], 0, 5) #corpse decay farm dist
            κ_d = TruncatedNormal(data.Immunity[1], data.Immunity[2], 0, 1)

            ζ[cs[i]+1:cs[i+1]] = rand(ζ_d,nt)
            γ[cs[i]+1:cs[i+1]] = rand(γ_d,nt)
            ω[cs[i]+1:cs[i+1]] = rand(ω_d,nt)
            ρ[cs[i]+1:cs[i+1]] = rand(ρ_d,nt)
            κ[cs[i]+1:cs[i+1]] = rand(κ_d,nt)

            μ_b_r = rand(μ_b_d,nt) 
            μ_b[cs[i]+1:cs[i+1]] = μ_b_r
            μ_d[cs[i]+1:cs[i+1]] = birth_death_mod*μ_b_r
            μ_c[cs[i]+1:cs[i+1]] = K[cs[i]+1:cs[i+1]]

            λ[cs[i]+1:cs[i]+nf] .= rand(λ_fd,nf)
            λ[cs[i]+nf+1:cs[i+1]] .= rand(λ_ld,nl)

        end
        
    end

    return  μ_b, μ_d, μ_c, ζ, γ, ω, ρ, λ, κ
    
end

function read_inputs(path)
    #=
    Function to read in the data for the tau simulation. Expecting a file for simulation meta data, 
    a folder with population data and another folder with seasonal data
    
    Inputs:
    -path, the path to directory with data for suns

    Outputs:
    -simulation, the simulation meta data
    -pops, data on n populations used in the model
    -seasonal, seaonal data on the n populations 
    =#

    Simulation = CSV.read("$(path)/Simulation_Data.csv", DataFrame; comment="#") #reading in simulation meta data
    
    n_inf  = infected_populations(Simulation) #what population is seeded with ASF
        
    Sim = Meta_Data(Simulation,n_inf)
    
    Pops = Vector{Population_Data}(undef, Sim.N_Pop)
    #Seasons = [DataFrame() for _ in 1:Sim.N_Pop] seasons not currently in use
    
    for i in 1:Sim.N_Pop
        pop_data = CSV.read("$(path)/Population/Population_$(i).csv", DataFrame; comment="#") 
        Pops[i] = Population_Data(pop_data)
        #Seasons[i] = CSV.read(string(path,"Seasonal/Seasonal_",i,".csv"), DataFrame; comment="#")
    end
    
    return Sim, Pops #, Seasons
    
end 

function population_connection(counts, sim)
    #=
     Function to build the the connections between the populations
     Could put in some error checking to make sure the entered populations are reasonable
     
     Inputs:
     - counts, vector of the amount of farms and feral groups in each population
     - sim, overall params will use to see if we need custom connections or not
     Outputs:
     -connections, vector of vectors containing the populations each population is connected too
     =#

    if (sim.C_Type == "o") & (counts.pop > 2) #custom input

        println("-------------------------------------")
        println("$(counts.pop) Populations!\nEnter connections for each population\n(for multiple seperate with space)")
        println("-------------------------------------")
            
        connections = Vector{Vector{Int}}(undef, counts.pop)

        for i in 1:counts.pop
            println("Population $(i) Connects to:")
            nums = readline()
            numv =  parse.(Int, split(nums, " "))
            connections[i] = numv
        end
    else #will have pre_loaded connections
        connections  = premade_connections(sim.C_Type, counts.pop)
    end    
    
    return connections
end

function premade_connections(type_c, Ni)
    
    #three different connection types, c-circular, l-line, t-total, will defualt to line 
    connections = Vector{Vector{Int}}(undef, Ni)
    
    if Ni == 2 #if only 2 populations have to be connected
        
        connections[1] = [2]
        connections[2] = [1]
        
    elseif type_c == "c" #circular connection
        
        for i in 1:Ni
            
            if i == 1
                connections[i] = [i+1, Ni]
            elseif i != Ni
                 connections[i] = [i-1,i+1]
            else
                connections[i] = [1, i-1]
            end
        end
     
    elseif type_c == "t" #every population connects to another
        
        for i in 1:Ni
            
            co = Vector(1:Ni)
            filter!(e->e≠i,co)
            connections[i] = co
            
        end
    else #line
        for i in 1:Ni
            if i == 1
                connections[i] = [i+1]
            elseif i != Ni
                connections[i] = [i-1,i+1]
            else
                connections[i] = [i-1]
            end
        end
    end
    
    return connections 
end

function infected_populations(input)
    number_pops  = parse(Int8, input.Value[4]) #number of populations 
    number_seeded = parse(Int8, input.Value[5]) #number of populations seeded with ASF
    con_type = input.Value[7]

    if (con_type == "o") & (number_pops > 1) #for runs with 2+ populations can choose what population we seed if custom
        println("-------------------------------------")
        println(string(number_pops," Populations!\nEnter Populations Seeded With ASF! \n(for multiple seperate with space)"))
        println("-------------------------------------") 
        nums = readline()
        numv =  parse.(Int, split(nums, " "))
    elseif (number_pops > 1) & (number_seeded > 1)  #if multiple populations and multiple seeded will randomly assigned seeded pops
        s_pops = shuffle!(MersenneTwister(1234), Vector(1:number_pops))
        numv = s_pops[1:number_seeded]
    else #1 seeded population
        numv = [1]
    end
    
    return numv

end 

function beta_construction(sim, pops, counts, network)
    #this function is to convert the base network we have created into a transmission coefficant matrix

    n_pops = counts.pop
    n_cs = counts.cum_sum
    beta = Float32.(copy(network))
    connected_births = copy(network)
    
    connected_births[connected_births .!= 200] .= 0 #only wanted connected groups within same pop
    connected_births =  connected_births .÷ 200 #setting to ones
    connected_pops = copy(connected_births)

    for i in 1:n_pops #iterating through

        data = pops[i]

        n_aim = data.N_int[1]

        connected_pops[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]] *= i


        beta_pop = beta[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]]
        
        if sim.Identical #no variation, just mean of dist
            beta_pop[beta_pop .== 100] .= data.B_f[1] #intra feral
            beta_pop[beta_pop .== 200] .= data.B_ff[1] .* (1/n_aim) #inter feral
            beta_pop[beta_pop .== 300] .= data.B_fl[1] #farm feral
            beta_pop[beta_pop .== 400] .= data.B_l[1] #intra farm
        else #from dist
            i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
            i_ff = TruncatedNormal(data.B_ff[1],data.B_ff[2],0,5) #inter group

            n_intra = length(beta_pop[beta_pop .== 100])
            n_inter = length(beta_pop[beta_pop .== 200])

            b_intra = rand(i_f, n_intra)
            b_inter = rand(i_ff, n_inter) .* (1/n_aim)

            beta_pop[beta_pop .== 100] = b_intra
            beta_pop[beta_pop .== 200] = b_inter  


            if counts.farm[i] > 0
                i_fl = TruncatedNormal(data.B_fl[1],data.B_fl[2],0,5)
                i_l = TruncatedNormal(data.B_l[1],data.B_l[2],0,5)

                n_farm_feral = length(beta_pop[beta_pop .== 300])
                n_farm = length(beta_pop[beta_pop .== 400])

                b_farm_feral = rand(i_fl, n_farm_feral)
                b_farm = rand(i_l, n_farm)

                beta_pop[beta_pop .== 300] = b_farm_feral
                beta_pop[beta_pop .== 400] = b_farm

            end

        end

        beta[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]] = beta_pop

    end

    return beta, connected_pops, connected_births

end

function build_populations(sim, pops, network, counts)
    #=
    Function to build the initial populations for each group in each population and Each group is divided into
    five classes- S,E,I,R,C. Also determines in which group ASF is seeded
    
    Inputs:
    - sim, the simulation meta data
    - pops, data on n populations used in the model

    Outputs:
    -y_total, vector of intial population distributions 
    -counts, vector of the amount of farms and feral groups in each population
    =#
    
    N_class = 5 #S,E,I,R,C
    N_pop = counts.pop
    n_cs = counts.cum_sum
    N_groups = n_cs[end]


    boar = 0.2 #percentage of groups that are solitary boar
    p_i = sim.N_Inf #what population seeded with ASF
    n_seed = sim.N_Seed #number of groups in said population that are seeded
    
    y_total = Vector{Int16}(undef, N_groups*N_class) #vector to store initial populations
    densities = Vector{Float32}(undef, N_pop) #vector to store each population's density
    areas = Vector{Float32}(undef,N_pop) #vector to store each population's area
    
    for i in 1:N_pop #looping through all populations
        
        #population data
        data = pops[i]
        
        #Density of population
        Density_D = TruncatedNormal(data.Dense[1],data.Dense[2],0,100)
        Density = rand(Density_D)
        densities[i] = Density
        
        #vector to store  population numbers for each class of each group
        y_pop = zeros(Int32,N_class*(n_cs[i+1]-n_cs[i]))
        
        
        N_feral = counts.feral[i] #number of feral groups 
        N_boar = round.(Int16, N_feral .* boar)#number of wild boar in pop
        N_sow = N_feral - N_boar  #number of sow groups in pop
        N_farm = counts.farm[i] #number of farms in population

        sow_dist = TruncatedNormal(data.N_f[1],data.N_f[2],3,500) #dist for number of pigs in sow feral group
        sow_groups = round.(Int16,rand(sow_dist, N_sow)) #drawing the populations of each feral group in population
    
        pop_network = copy(network[n_cs[i]+1:n_cs[i+1]-N_farm,n_cs[i]+1:n_cs[i+1]-N_farm]) #isolating network for this feral pop 
        pop_network[pop_network .!= 0] .= 1 #seeting all 
        group_degree = vec(sum(Int16, pop_network, dims = 2)) .- 1 #group degree -1 as not counting inta group connections 

        if sum(group_degree  .== 0) > 0
            println("Warning ",sum(group_degree  .== 0), " disconnected feral groups!")
        end

        #now want to choose the N_boar groups with the highest degree as these are boars, the others will be sow_dist
        index_boar = sort(partialsortperm(group_degree,1:N_boar,rev=true)) #index of all boar groups
        index_sow = setdiff(1:N_feral, index_boar) #index of all sow groups
        index_sow_pop = N_class*(index_sow .- 1) .+ 1 #index of S for all sows
        index_boar_pop = N_class*(index_boar .- 1) .+ 1 #index of S for all boars

        if i in p_i #population that have ASF in a group

            group_index = N_class*((1:N_feral) .- 1) .+ 1 #index of S for all groups
            seeded_groups = unique(rand(group_index,n_seed)) #index of S for all groups that are seeded with ASF!

            sow_inf = seeded_groups[seeded_groups .∈  [index_sow_pop]] #sow groups that are seeded
            boar_inf = seeded_groups[seeded_groups .∈  [index_boar_pop]] #boar groups that are seeded

            n_si = length(sow_inf)
           
            #Disease Free Groups
            disease_free_sows = setdiff(index_sow_pop,sow_inf)
            disease_free_boars = setdiff(index_boar_pop,boar_inf)
            
        
            y_pop[disease_free_sows] = sow_groups[n_si+1:end]
            y_pop[disease_free_boars] .= 1 

            #Now the dieased group(s)!
            
            #for sows
            d_e = TruncatedNormal(data.N_e[1],data.N_e[2], 1, 100) #dist for number of pigs exposed
            d_i = TruncatedNormal(data.N_i[1],data.N_i[2], 1, 100) #dist for number of pigs infected
                    
            t_e = round.(Int16,rand(d_e,n_si)) #number of exposed in infected sow groups
            t_i = round.(Int16,rand(d_i,n_si)) #number of infected in infected sow groups
                    
            t_pop = sow_groups[1:n_si] #expected total population of infected sow groups

            y_pop[sow_inf] = max.(0,t_pop-t_e-t_i) #S
            y_pop[sow_inf .+ 1] = t_e #E
            y_pop[sow_inf .+ 2] = t_i #I

            #for boars
            #I am assuming that all boars will be infective

            y_pop[boar_inf .+ 2] .= 1 
            
        else # population does not have ASF in any groups
            
            y_pop[index_sow_pop] = sow_groups
            y_pop[index_boar_pop] .= 1
            
        end
        
        total_pop = sum(y_pop)
        total_area = total_pop/Density
        
        areas[i] = total_area
        
        #livestock populations
        
        if N_farm > 0
            group_farm_pops = TruncatedNormal(data.N_l[1],data.N_l[2],1,10000)#distribution of group sizes         
            farm_groups = round.(Int,rand(group_farm_pops, N_farm))
            y_pop[N_class*N_feral+1:5:end] = farm_groups
        end
        
        n_cs_class = n_cs*N_class
        
        y_total[n_cs_class[i]+1:n_cs_class[i+1]] = y_pop
        
    end
   
    #updating area and density storage as we have now caluclated them
    counts.density = densities
    counts.area = areas
    return trunc.(Int,y_total)
    
end

end
