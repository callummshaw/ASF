module ASF_Inputs

using LinearAlgebra
using DelimitedFiles
using DataFrames
using CSV
using FileIO
using LinearAlgebra
using NPZ
using Random, Distributions
using LightGraphs

export Model_Data

abstract type Data_Input end

struct Meta_Data <: Data_Input
    #=
    Structure to store all the meta data infomation for the run:
    number of years, if the model is being run in ensemble, if
    using distros or just mean values for parameters, the number of 
    populations and the number of said populations seeded with ASF
    =#
    years::Float64
    N_ensemble::Int64
    Identical::Bool
    N_Pop::Int64
    N_Inf::Vector{Int64}
    
    function Meta_Data(input, numv)
        Ny = parse(Int64, input.Value[1])
        Ne = parse(Float64,input.Value[2])
        I = (input.Value[3] == "true")
        Np = parse(Int64, input.Value[4])
        Ni = numv
        
        new(Ny,Ne,I,Np,Ni)
    end
end

struct Population_Data <: Data_Input
    
    Dense::Vector{Float64} #density of population
    N_feral::Vector{Int64} #number of feral groups
    N_farm::Vector{Int64} #number of farm groups
    N_int::Vector{Int64} #average interconnection between feral groups
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
        R = [input.Mean[10],input.STD[10]]
        L = [input.Mean[11],input.STD[11]]
        C = [input.Mean[12],input.STD[12]]
        Dl = [input.Mean[13],input.STD[13]]
        Df = [input.Mean[14],input.STD[14]]
        Npf = [input.Mean[15],input.STD[15]]
        Npl = [input.Mean[16],input.STD[16]]
        Npe = [input.Mean[17],input.STD[17]]
        Npi = [input.Mean[18],input.STD[18]]
        B = [input.Mean[19],input.STD[19]]
        Dn = [input.Mean[20],input.STD[20]]
        
        new(Den, Nf, Nl, Ni ,Bf, Bl, Bff, Bfl, D, R, L, C, Dl, Df, Npf, Npl, Npe, Npi, B, Dn)
        
    end
    
end

struct Population_Breakdown
    #=
    Structure to store key data on each population
    =#
    feral::Vector{Int64}
    farm::Vector{Int64}
    
    density::Vector{Float64}
    area::Vector{Float64}

    
    pop::Int64
    total::Vector{Int64}
    cum_sum::Vector{Int64}
    
    inf::Vector{Int64}
    
    function Population_Breakdown(feral,farm, density,area, numv)
        n = size(feral)[1]
        t = feral + farm
        cs = pushfirst!(cumsum(t),0)
        
        new(feral,farm,density,area,n,t,cs,numv)
    end
end 

mutable struct Model_Parameters
    #=
    Structure to store key parameters
    =#
    
    β::Matrix{Float64} #transmission matrix
    β_b::Matrix{Float64} #what feral groups are linked to each other, for births
    β_d::Matrix{Float64} #used to identify feral groups and farms for each population for density calcualtions
    
    μ_b::Vector{Float64} #birth rate
    μ_d::Vector{Float64} #natural death rate
    μ_g::Vector{Float64} #density dependent death/birth rate (to ensure carrying capicity)
    
    ζ::Vector{Float64} #latent rate
    γ::Vector{Float64} #recovery rate
    ω::Vector{Float64} #corpse infection modifier
    ρ::Vector{Float64} #death probability
    λ::Vector{Float64} #corpse decay rate
    
    Populations::Population_Breakdown #breakdown of population
    
    function Model_Parameters(sim, pops, U0, Populations)
        
        β, β_density = population_beta(sim, pops, Populations)
        β_connections = migration_births(β, Populations)
        
        μ_birth, μ_death, μ_capicty, ζ, γ, ω, ρ, λ = parameter_build(sim, pops, U0, Populations)
        
        new(β, β_connections, β_density, μ_birth, μ_death, μ_capicty, ζ, γ, ω, ρ, λ, Populations)
    end
    
end

struct Model_Data
    #=
    Structure to store key data on mode
    =#
    Time::Tuple{Float64, Float64} #Model run time
    U0::Vector{Int64} #Initial Population
    Parameters::Model_Parameters #Model parameters
    Populations_data::Vector{Population_Data} #distributions for params
    function Model_Data(Path)
        sim, pops = read_inputs(Path)
        
        Time = (0.0,sim.years[1]*365)
        
        U0, counts = intial_group_pops(sim, pops) #initial populations
        
        Parameters = Model_Parameters(sim, pops, U0, counts)
        
        new(Time, U0, Parameters, pops)
        
    end
    
end

function parameter_build(sim, pops, init_pops, counts)
    #=
    Function that builds most parameters for model
    =#
   
    K = init_pops[1:5:end] + init_pops[2:5:end] + init_pops[3:5:end] #carrying capacity of each group
    
    # All other params
    
    ζ = [] #latent rate
    γ = [] #recovery/death rate
    μ_b = [] #births
    μ_d = [] #natural death rate
    μ_g = [] #density dependent deaths
    ω = [] #corpse infection modifier
    ρ = [] #ASF mortality
    λ = [] #corpse decay rate
    
    for i in 1:counts.pop
        data =  pops[i]
        
        nf = counts.feral[i]
        nl = counts.farm[i]
        nt = counts.total[i]
        
        cs = counts.cum_sum
        
        if sim.Identical == true #if running off means
            
            ζ_r = repeat([data.Latent[1]],nt)
            append!(ζ,ζ_r)
            
            γ_r = repeat([data.Recovery[1]],nt)
            append!(γ,γ_r)
            
            μ_b_r = repeat([data.Birth[1]],nt)
            append!(μ_b,μ_b_r)
            
            μ_d_r = repeat([data.Death_n[1]],nt)
            append!(μ_d,μ_d_r)
            
            μ_g_r =  (μ_b_r-μ_d_r)./K[cs[i]+1:cs[i+1]]
            append!(μ_g,μ_g_r)
            
            ω_r = repeat([data.Corpse[1]],nt)
            append!(ω,ω_r)
            
            ρ_r = repeat([data.Death[1]],nt)
            append!(ρ,ρ_r)
            
            λ_fr = repeat([data.Decay_f[1]],nf)
            λ_lr = repeat([data.Decay_l[1]],nl)
            append!(λ,λ_fr)
            append!(λ,λ_lr)

            
        else #running of distros
            
            ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2],0,5) #latent dist
            γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2],0,5) #r/d rate dist
            μ_b_d = TruncatedNormal(data.Birth[1], data.Birth[2],0,1) #birth dist
            μ_d_d = TruncatedNormal(data.Death_n[1], data.Death_n[2],0,1) #n death dist
            ω_d = TruncatedNormal(data.Corpse[1], data.Corpse[2],0,1) #corpse inf dist
            ρ_d = TruncatedNormal(data.Death[1], data.Death[2],0,1) #mortality dist
            λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2],0,1) #corpse decay feral dist
            λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2],0,5) #corpse decay farm dist

            append!(ζ,rand(ζ_d,nt))
            append!(γ,rand(γ_d,nt))
            append!(ω,rand(ω_d,nt))
            append!(ρ,rand(ρ_d,nt))

            μ_b_r = rand(μ_b_d,nt)
            μ_d_r = rand(μ_d_d,nt)
            
            append!(μ_b, μ_b_r)
            append!(μ_d, μ_d_r) 
            
            μ_g_r =  (μ_b_r-μ_d_r)./K[cs[i]+1:cs[i+1]]
            append!(μ_g,μ_g_r)
            
            append!(λ,rand(λ_fd,nf))
            append!(λ,rand(λ_ld,nl))
        
        end
        
    end

    return  μ_b, μ_d, μ_g, ζ, γ, ω, ρ, λ
    
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
    
    Simulation = CSV.read(string(path,"Inputs/Simulation_Data.csv"), DataFrame; comment="#") #reading in simulation meta data
    
    
    n_inf  = infected_populations(Simulation)
        
    Sim = Meta_Data(Simulation,n_inf)
    
    Pops = Vector{Population_Data}(undef, Sim.N_Pop)
    #Seasons = [DataFrame() for _ in 1:Sim.N_Pop] seasons not currently in use
    
    for i in 1:Sim.N_Pop
        pop_data = CSV.read(string(path,"Inputs/Population/Population_",i,".csv"), DataFrame; comment="#") 
        Pops[i] = Population_Data(pop_data)
        #Seasons[i] = CSV.read(string(path,"Seasonal/Seasonal_",i,".csv"), DataFrame; comment="#")
    end
    
    return Sim, Pops #, Seasons
    
end 

function migration_births(β, counts)
   #=
    function for to find connected groups to determine births from other groups that migrate into said group
    needed to help prevent stochastic group die-out
    =#
    
    βb = β .!= 0 #matrix of all connected groups
    βb[diagind(βb)] .= 0 #setting intra group value to 0,as meant to be births from other groups
    
    migration_modifier = 0.01 # X% external births vs internal births
    
    for j in 1:counts.pop #here to stop migratory births from connected groups in neighbouring populations
        
        ll = counts.cum_sum[j]+counts.feral[j] + 1
        uu = counts.cum_sum[j+1]
        
        βb[:,ll:uu] .= 0
        βb[ll:uu,:] .= 0
            
    end
    
    return βb
end

function population_connection(counts)
    #=
     Function to build the the connections between the populations
     Could put in some error checking to make sure the entered populations are reasonable
     
     Inputs:
     - counts, vector of the amount of farms and feral groups in each population
     
     Outputs:
     -connections, vector of vectors containing the populations each population is connected too
     =#
     println("-------------------------------------")
     println(string(counts.pop," Populations!\nEnter connections for each population\n(for multiple seperate with space)"))
     println("-------------------------------------")
         
     connections =  [Vector() for _ in 1:counts.pop]
     
     for i in 1:counts.pop
         println(string("Population ", i , " Connects to:"))
         nums = readline()
         numv =  parse.(Int, split(nums, " "))
         connections[i] = numv
     end
     
     return connections
end


function infected_populations(input)
    
    Ni = parse(Int64, input.Value[4])

    if Ni == 1
        #println("-------------------------------------")
        #println("Single Population!\nThe Only Population is Seeded With ASF!")
        #println("-------------------------------------") 
        numv = [1]
    else
        println("-------------------------------------")
        println(string(Ni," Populations!\nEnter Populations Seeded With ASF! \n(for multiple seperate with space)"))
        println("-------------------------------------") 
        nums = readline()
        numv =  parse.(Int, split(nums, " "))
        
    end
    
    return numv

end 

function population_beta(sim, pops, counts)
    #=
    function used to construct the transmission co-efficant matrix over all populations,
    will build each population individaully then combine at end if need. 
    =#
    
    N_pops = sim.N_Pop
    
    beta_p = Matrix{Float64}[]
    beta_dens = Matrix{Float64}[]
    
    for x in 1:N_pops 
        
        data = pops[x]
        
        nf =  trunc(Int64,counts.feral[x]) #number of feral groups within pop
        nl = trunc(Int64,counts.farm[x]) #number of farms within pop
        
        n_t = nf + nl
        
        n_o = nf -1 #number of other feral groups; excluding "target group"

        if data.N_int[1] > n_o
           # @warn "Warning interconnectedness exceeds number of feral groups in population; average enterconnectedness set to total  number of feral groups - 1"
            n_aim = n_o 
        else
            n_aim = data.N_int[1]
        end

        #using an erdos-renyi random network to determine inter-group interactions
        p_c = n_aim / n_o #probability of a connection

        if p_c == 1
            beta_feral = erdos_renyi(nf,sum(1:n_o)) #every group connected with every other
        else
            beta_feral = erdos_renyi(nf,p_c) #not all groups are connected to every other
        end

        beta_m = Matrix(adjacency_matrix(beta_feral))
        
        beta_m = Float64.(beta_m) 
        
        beta_t = (x+1)*beta_m #used to keep track of density
        #looking at feral
        
        #setting feral values
        if sim.Identical == true #if no varition between groups in a pop

            beta_m = beta_m * data.B_ff[1] .* (1/n_aim) #inter group
            beta_m[diagind(beta_m)] .= data.B_f[1] #intra group

        else #variaiton between inter-population groups

            i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
            b_i_f = rand(i_f, nf) 
            beta_m[diagind(beta_m)] = b_i_f
            
            i_ff = TruncatedNormal(data.B_ff[1],data.B_ff[2],0,5) #inter group
            
            for i in 1:nf
                for j in 1:nf
                    if i>j && beta_m[i,j] != 0
                        beta_m[i,j] = beta_m[j,i]  = rand(i_ff).* (1/n_aim) 
                    end
                end
            end

        end
        
        

        #now loooking at farms
        if nl != 0  #there are farm populations 
            
            N = nf + nl
            beta = zeros((N,N))
            beta_tt = zeros((N,N))
            beta[1:nf,1:nf] = beta_m 
            beta_tt[1:nf,1:nf] = beta_t

            #currently assumes 1 farm-group interaction term

            if sim.Identical == true #if no varition between groups in a pop

                for i in nf+1:N

                    feral_pop = rand(1:nf) # the feral population the farm can interact within
                    beta[i,i] = data.B_l[1] #transmission within farm pop

                    beta[i,feral_pop] = data.B_fl[1]
                    beta[feral_pop, i] = data.B_fl[1]
                    
                    beta_tt[i,feral_pop] = 99
                    beta_tt[feral_pop, i] = 99
                end

            else
                
                i_l = TruncatedNormal(data.B_l[1],data.B_l[2],0,5)
                i_fl = TruncatedNormal(data.B_fl[1],data.B_fl[2],0,5)

                for i in nf+1:N

                    feral_pop = rand(1:nf) # the feral population the farm can interact within
                    beta[i,i] = rand(i_l)#transmission within farm pop

                    b_i_fl = rand(i_fl)
                    beta[i,feral_pop] = b_i_fl
                    beta[feral_pop, i] = b_i_fl
                    
                    beta_tt[i,feral_pop] = 99
                    beta_tt[feral_pop, i] = 99
                end

            end
        
        else #no farms in this population
            beta = beta_m 
            beta_tt = beta_t
    
        end
        
        push!(beta_p, beta)
        push!(beta_dens, beta_tt)
    end
    

    if N_pops > 1
        beta,beta_density = combine_beta!(beta_p,beta_dens,counts)
    else
        beta = beta_p[1]
        beta_density = beta_dens[1]
    end
    
    return beta, beta_density
    
end


function combine_beta!(beta_p, beta_d, counts)
    
    connections = population_connection(counts)
    
    cs_g = counts.cum_sum
  
    N = sum(counts.total)
    beta = zeros(N,N)
    beta_density  = zeros(N,N)
    
    links = []
    
        for i in 1:counts.pop

            linked_pops = connections[i]
            beta[cs_g[i]+1:cs_g[i+1],cs_g[i]+1:cs_g[i+1]] = beta_p[i]
            beta_density[cs_g[i]+1:cs_g[i+1],cs_g[i]+1:cs_g[i+1]] = beta_d[i]
            for j in connections[i]
                
                new_link = sort([i,j]) #checking if we have already linked the two populations

                if new_link ∉ links
                    #println(new_link)
                    push!(links, sort([i,j])) #storing the link

                    println("-------------------------------------")
                    println(string("Strength of Population ", i, " to Population ", j, " Transmission:"))
                    str = readline()
                    str = parse(Float64, str) 

                    #the chosen population
                    ll = cs_g[i] + 1
                    ul = cs_g[i] + counts.feral[i]
                    l_group = rand(ll:ul)

                    #populations the chosen population is linked too
                    ll_s = cs_g[j] + 1

                    ul_s = cs_g[j] + counts.feral[j]

                    s_group = rand(ll_s:ul_s)

                    beta[l_group,s_group] = str
                    beta[s_group,l_group] = str
                    
                end

            end
        end
    
    beta_density[diagind(beta_density)] .= 0
    
    
    return beta, beta_density
end

function intial_group_pops(sim, pops)
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
    
    p_i = sim.N_Inf #what population seeded with ASF
    y_total = [] #vector to store initial populations
    
    farm_count = [] #vector to store farm counts
    feral_count = [] #vector to store feral counts
    
    densities = [] #vector to store each population's density
    areas = [] #vector to store each population's area
    
    for i in 1:sim.N_Pop #looping through all populations
        
        data = pops[i]
        
        #Density of population
        Density_D = TruncatedNormal(data.Dense[1],data.Dense[2],0,100)
        Density = rand(Density_D)
        
        #number of farms and feral groups for each population are drawn from a distribution
        
        if data.N_feral[1] == 0
            N_feral = 0
            push!(feral_count, N_feral)
        else
            N_feral_D = TruncatedNormal(data.N_feral[1],data.N_feral[2],0,1000) #number of feral group distribution
            N_feral = trunc(Int64,rand(N_feral_D))
            push!(feral_count, N_feral)
        end
       
        
        if data.N_farm[1] == 0
            N_farm = 0
            push!(farm_count, N_farm)
        else
            N_farm_D = TruncatedNormal(data.N_farm[1],data.N_farm[2],0,100) #number of farms distribution
            N_farm =  trunc(Int64,rand(N_farm_D))
            push!(farm_count,N_farm)
        end
        
        push!(densities, Density)
        
        N_total = N_feral + N_farm
        
        y0 = zeros(N_class*N_total)
        
        group_feral_pops = Normal(data.N_f[1],data.N_f[2]) #dist for number of pigs in selected feral group
        pop_groups = round.(Int,rand(group_feral_pops, N_feral)) #drawing the populations of each feral group in population
        
        #now seededing ASF in feral populations
        
        if i in p_i #population that have ASF in a group
            
            g_i = rand(1:N_feral) #choosing group to seed ASF
         
            for k in 1:N_feral #looping through groups
                
                l = (k-1)*N_class+1
                
                if k == g_i # if group is the seeded group
                    
                    d_e = TruncatedNormal(data.N_e[1],data.N_e[2], 0, 100) #dist for number of pigs exposed
                    d_i = TruncatedNormal(data.N_i[1],data.N_i[2], 0, 100) #dist for number of pigs infected
                    
                    t_e = round(Int,rand(d_e)) #number of exposed in group
                    t_i = round(Int,rand(d_i)) #number of infected in group
                    
                    t_pop = pop_groups[k]
                    
                    y0[l] = max(0,t_pop-t_e-t_i)
                    y0[l+1] = t_e
                    y0[l+2] = t_i
                    
                else
                    
                    y0[l] = pop_groups[k]
                    
                end
                
            end
            
        else # population does not have ASF in any groups
            
            y0[1:5:N_class*N_feral] = pop_groups
            
        end
        
        total_pop = sum(y0)
        total_area = total_pop/Density
        
        push!(areas, total_area)
        
        #livestock populations
        group_farm_pops = TruncatedNormal(data.N_l[1],data.N_l[2],0,10000)#distribution of group sizes         
        farm_groups = round.(Int,rand(group_farm_pops, N_farm))
        y0[N_class*N_feral+1:5:end] = farm_groups

        append!(y_total, y0)
        
    end
    
    counts = Population_Breakdown(feral_count,farm_count, densities, areas, sim.N_Inf)

    return trunc.(Int,y_total), counts

    
end


end
