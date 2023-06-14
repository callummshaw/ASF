"Module that builds the network used in the ASF model (M3 only)"
module Network

using Distributions
using Graphs
using LinearAlgebra
using Random

export build
export Network_Data


mutable struct Network_Data
    #=
    Structure to store key data on each population, internal just to keep track of a few params
    =#
    feral::Vector{Int16} #number of feral in each pop
    farm::Vector{Int8} #number of farms in each pop
    pop::Int8 #numbr of populations!
    total::Vector{Int16} #the total populations in said region
    cum_sum::Vector{Int16}
    
    inf::Int8
    
    density::Vector{Float32}
    area::Vector{Float32}

    inter_connections::Vector{Matrix{Int16}}
    pop_connections::Matrix{Int16}
    
    networks::Vector{Matrix{Int16}}
    function Network_Data(feral,farm, inf, density,area, inter_con, networks)
        n = size(feral)[1]
        t = feral + farm
        cs = pushfirst!(cumsum(t),0)

        new(feral,farm,n,t,cs, inf, density, area,inter_con,zeros(Int16,2,2), networks)
    end
end 



function build(sim, pops, verbose, pop_net)

    #This function builds the network! (trivial for M1 and M2)
    n_p = sim.N_Pop # number of populations per region
    n_r = size(pops)[1] #number of regions
    n_pops = n_p*n_r #total number of populations
    network = Vector{Matrix{Int16}}(undef, n_pops) #vector to store all the connection matrices
    network_con = Vector{Matrix{Int16}}(undef, n_pops)
    feral_pops = Vector{Int16}(undef, n_pops)
    farm_pops = Vector{Int16}(undef, n_pops)
    if sim.Model == 3 #model 3 so need to generate network!
        for pop in 1:n_pops #looping through all N populations
            
            data_i = (pop-1) ÷ n_p + 1 #get correct index for population

            feral_max = 5000 #maximum number of feral groups, will run very slow near maximum
            farm_max = 100 #maximum number of farm

            data = pops[data_i]

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
        
            network_feral = Matrix(adjacency_matrix(feral))

            if nl != 0  #there are farm populations 
                
                N = nf + nl
                network_combined = zeros((N,N))
                network_combined[1:nf,1:nf] = network_feral 
            
                #currently assumes 1 farm-group interaction term

                for i in nf+1:N
                    feral_pop = rand(1:nf) # the feral population the farm can interact within
                    network_combined[i,i] = 1 

                    network_combined[i,feral_pop] = 1 
                    network_combined[feral_pop, i] = 1 

                end
                
                ncu = UpperTriangular(network_combined)
                nc = getindex.(findall(>(0),ncu), [1 2])

                network[pop] = network_combined
                network_con[pop] = nc  

            else #no farms in this population
                network[pop] = network_feral
                ncu = UpperTriangular(network_feral)
                nc = getindex.(findall(>(0),ncu), [1 2])
                network_con[pop] = nc
            end
            
        end
    else #model 1 or 2 so no network structure! Only 1 population aswell!

        feral_pops[1] = 1 #only 1 large group!
        farm_pops[1] = 0 #no farms(yet...)
        network[1] = [1][:,:]
        network_con[1] = zeros(2,2)
    end
    
    counts = Network_Data(feral_pops,farm_pops, sim.N_Seed,[0.0],[0.0], network_con, network)

    if (n_pops > 1) & (sim.Model == 3) #model 3 with more than 1 population! Therefore need to connect!
        counts.pop_connections = combine_networks(network,sim,counts,pop_net, verbose) #connections between networks
    end

    return counts

end

function combine_networks(network,sim, counts, pop_net,verbose)
    #we have generated all the networks, but they need to be combined!

    N_connections = sim.N_Con #number of connecting groups between populations
    n_pops = counts.pop
    n_cs = counts.cum_sum

    if pop_net isa Int64
        if verbose
            @warn "Must input network structure, defaulting to line!"
        end
        pop_matrix =  UpperTriangular(Matrix(adjacency_matrix(path_graph(n_pops))))
    else
        pop_matrix =  UpperTriangular(Matrix(adjacency_matrix(pop_net))) #matrix of our meta-population network
    end
  
    #now we need to connect all the networks with each other!
    if size(pop_matrix)[1] != n_pops
        @warn "Population level network and number of populations generated from input do not match, defaulting to line!"
        pop_matrix =  UpperTriangular(Matrix(adjacency_matrix(path_graph(n_pops))))
    end
    
    connections = findall(>(0),pop_matrix) #vector with all our connections
    
    all_nodes = zeros(Int16,0) 
    
    S1 = zeros(Int16,0) 
    S2 = zeros(Int16,0)
    S1P = zeros(Int16,0) 
    S2P = zeros(Int16,0)
    
    for i in connections
        p1 = i[1]
        p2 = i[2]

        p1_base = n_cs[p1]
        p2_base = n_cs[p2]
        
        N1_nodes = 0
        while N1_nodes == 0      
            N1_nodes = find_nodes(network[p1], N_connections, all_nodes) #nodes from pop 1 
        end
        
        N2_nodes = 0
        while N2_nodes == 0
            N2_nodes = find_nodes(network[p2], N_connections, all_nodes) #nodes from pop 2
        end
        
        append!(all_nodes,N1_nodes)
        append!(all_nodes,N2_nodes)

        append!(S1,N1_nodes)
        append!(S2,N2_nodes)

        append!(S1P,repeat([p1],length(N1_nodes)))
        append!(S2P,repeat([p2],length(N1_nodes)))
    end 

    SS_Nodes = zeros(Int16,length(S1),4)
    SS_Nodes[:,1] = S1
    SS_Nodes[:,2] = S2
    SS_Nodes[:,3] = S1P
    SS_Nodes[:,4] = S2P

    return SS_Nodes

end

function find_nodes(Network, N_connections, all_nodes)
        
    p1g = zeros(Int16,0) #groups we are using
    p1c = zeros(Int16,0) #centre groups that we search for links from (subet of p1g)

    g_p = rand(1:size(Network)[1]) #base group of p1 that is doing the connecting! now need to find n neighbours

    while g_p in all_nodes #making sure starting element is not used in anyother connections
        g_p = rand(1:size(Network)[1])
    end 

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
    
    return shuffle(p1g)

end

end