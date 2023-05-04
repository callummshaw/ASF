module Network

using Distributions
using Graphs
using LinearAlgebra

export build
export Network_Data


mutable struct Network_Data
    #=
    Structure to store key data on each population, internal just to keep track of a few params
    =#
    feral::Vector{Int16}
    farm::Vector{Int8}
    pop::Int8
    total::Vector{Int16}
    cum_sum::Vector{Int16}
    
    inf::Int8
    
    density::Vector{Float32}
    area::Vector{Float32}

    function Network_Data(feral,farm, inf, density,area)
        n = size(feral)[1]
        t = feral + farm
        cs = pushfirst!(cumsum(t),0)
        

        new(feral,farm,n,t,cs, inf, density, area)
    end
end 



function build(sim, pops, cn, verbose)

    #This function builds the network! (trivial for M1 and M2)

    n_pops = sim.N_Pop #number of populations
    network = Vector{Matrix{Int16}}(undef, n_pops) #vector to store all the 
    feral_pops = Vector{Int16}(undef, n_pops)
    farm_pops = Vector{Int16}(undef, n_pops)

    if sim.Model == 3 #model 3 so need to generate network!
        for pop in 1:n_pops 
        
            feral_max = 5000 #maximum number of feral groups, will run very slow near maximum
            farm_max = 100 #maximum number of farm

            data = pops[pop]

            #Feral dist
            if data.N_feral[1] == 0
                nf = 0
            else
                nf_d = TruncatedNormal(data.N_feral[1],data.N_feral[2],0,feral_max) #number of feral group distribution
                nf = trunc(Int16,rand(nf_d))
            end
        
            #Farm dist     
            if data.N_farm[1] == 0
                nl = 0
            else
                nl_d = TruncatedNormal(data.N_farm[1],data.N_farm[2],0,farm_max) #number of farms distribution
                nl =  trunc(Int16,rand(nl_d))
            end
            
            if verbose
                @info "$nf Feral Groups"
                @info "$nl Farm Populations"
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

        
            #this is where we build the three types of network, default is random but can allow for other network types
            if sim.Network == "s" #only works with even degrees 
                
                if verbose & isodd(n_aim)
                    @warn "Odd group degree detected, Scale free and small worlds require even degree"
                end
                adds = n_aim ÷ 2

                feral = barabasi_albert(nf, adds)

            elseif sim.Network == "w" #only works with even degrees
                
                if verbose & isodd(n_aim)
                    @warn "Odd group degree detected, Scale free and small worlds require even degree"
                end
                
                if cn != 0 
                    @warn "Custom rewiring probability being used!"
                    
                    if (cn > 1) | (cn < 0 )
                        @warn "Rewiring must be between 0 and 1"
                    end
                    rewire  = cn
                else
                    rewire = sim.N_param
                end
                
                feral = watts_strogatz(nf, n_aim, sim.N_param)
                
            else
                #using an erdos-renyi random network to determine inter-group interactions
                p_c = n_aim / n_o #probability of a connection

                if p_c == 1
                    feral = erdos_renyi(nf,sum(1:n_o)) #every group connected with every other
                else
                    feral = erdos_renyi(nf,p_c) #not all groups are connected to every other
                end
            end
            

            network_feral = Matrix(adjacency_matrix(feral))*200 #inter feral = 200
            network_feral[diagind(network_feral)] .= 100 #intra feral = 100

            if nl != 0  #there are farm populations 
                
                N = nf + nl
                network_combined = zeros((N,N))
                network_combined[1:nf,1:nf] = network_feral 
            
                #currently assumes 1 farm-group interaction term

                for i in nf+1:N
                    feral_pop = rand(1:nf) # the feral population the farm can interact within
                    network_combined[i,i] = 400 #transmission within farm pop = 400

                    network_combined[i,feral_pop] = 300 #feral-farm = 300
                    network_combined[feral_pop, i] = 300 #feral-farm = 300

                end
            
                network[pop] = network_combined

            else #no farms in this population
                network[pop] = network_feral
            end
            
        end
    else #model 1 or 2 so no network structure! Only 1 population aswell!

        feral_pops[1] = 1 #only 1 large group!
        farm_pops[1] = 0 #no farms(yet...)
        network[1] = [1][:,:]
        
    end

    counts = Network_Data(feral_pops,farm_pops, sim.N_Inf,[0.0],[0.0])
    
    if (n_pops > 1) & (sim.Model == 3)
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


end