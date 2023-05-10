






function analyse_out(input, counts)
    
    classes, t_steps, n_ens = size(input) 
    
    nft_alive = Vector{Vector{Float64}}(undef, n_ens)
    nft_exposed = Vector{Vector{Float64}}(undef, n_ens)
    
    
    nlt_alive = Vector{Vector{Float64}}(undef, n_ens)
    nlt_exposed = Vector{Vector{Float64}}(undef, n_ens)
    
    for i in 1:n_ens
        
        sol = input[i]
        
        data = reduce(vcat,transpose.(sol.u))

        if any(x->x==-1, data) == true
            println("Need to Reduce Timestep")
            
            #break
        end

        s_d = data[:,1:5:end]
        e_d = data[:,2:5:end]
        i_d = data[:,3:5:end]
        r_d = data[:,4:5:end]
        c_d = data[:,5:5:end]

        np = counts.pop

        nf_alive = Vector{Float64}(undef, np)
        nf_exposed = Vector{Float64}(undef, np)

        pl_alive = Vector{Float64}(undef, np)
        pl_exposed = Vector{Float64}(undef, np)


        for j in 1:np

            #FERAL
            
            ll = counts.cum_sum[j]+1
            uu = counts.cum_sum[j] + counts.feral[j]

            #disease-free classes
            s_j = s_d[:,ll:uu]
            r_j = r_d[:,ll:uu]

            final = s_j[end,:]+r_j[end,:]
            na = count(x->x!=0,final)

            nf_alive[j] = 100* na/counts.feral[j]

             #disease classes
            e_j = e_d[:,ll:uu]
            i_j = i_d[:,ll:uu]
            c_j = c_d[:,ll:uu]

            dis = e_j + i_j +c_j 
            ndis = sum(dis,dims=1)
            ne = count(x->x!=0,ndis)
            nf_exposed[j] = 100*ne/counts.feral[j]
            
            #FARM
            
            l_l = uu +1 
            u_l = counts.cum_sum[j+1]

            #disease-free classes
            s_j = s_d[:,l_l:u_l]
            r_j = r_d[:,l_l:u_l]

            final = s_j[end,:]+r_j[end,:]
            na = count(x->x!=0,final)

            pl_alive[j] = 100* na/counts.farm[j]

             #disease classes
            e_j = e_d[:,l_l:u_l]
            i_j = i_d[:,l_l:u_l]
            c_j = c_d[:,l_l:u_l]

            dis = e_j + i_j +c_j 
            ndis = sum(dis,dims=1)
            ne = count(x->x!=0,ndis)
            pl_exposed[j] = 100*ne/counts.farm[j]

        end
        
        nft_alive[i] = nf_alive
        nft_exposed[i] = nf_exposed
        
        nlt_alive[i] = pl_alive
        nlt_exposed[i] = pl_exposed
        
    end
    
    println("------------------------------------------------")
    println("Run Stats:")
    println("$(counts.pop) Populations")
    println("$(n_ens) Runs")
    println("------------------------------------------------")
    println("% of groups exposed to ASF in each population:")
    println(mean(nft_exposed))
    println("------------------------------------------------")
    println("% of groups that survive in each population:")
    println(mean(nft_alive))
    println("------------------------------------------------")
    if sum(counts.farm) > 0 
        println("% of farms exposed to ASF in each population:")
        println(mean(nlt_exposed))
        println("------------------------------------------------")

        println("% of farms that survive in each population:")
        println(mean(nlt_alive))
    end 
    
    #return mean(nft_alive), mean(nft_exposed), mean(nlt_alive), mean(nlt_exposed)

end

function save_raw_output(output, data, path, name)
    #=
    Function to save output
    =#
    
    dims = length(size(output))
    
    if dims == 3 #ensemble run
        classes, t_steps, n_ens = size(output)
    else #single run
        classes, t_steps = size(output)
        n_ens = 1
    end
   
    input_folder = splitpath(path)[end]
    base_path = rsplit(path, input_folder)[1]

    #making save directory
    dir = "$(base_path)Results/$(name)"

    println(dir)
    println(base_path)

    isdir(dir) || mkdir(dir)
    cd(dir)

    #copying inputs
    cp("$(path)", "Inputs",force = true)
    
    #saving run parameters
    save_object("parameters.jdl2", data)
    
    #saving output
    if dims == 3 #ensemble
        for i = 1:n_ens
            solnew = vcat(output.u[i].t',output[:,:,i])' #data
            writedlm("ASF_$(i).csv",  solnew, ',')
        end
    else
        solnew = vcat(sol.t',sol[:,:,i])'
        writedlm( "ASF_single",  solnew, ',')
    end
    
    
end




function three_statistics(data, threshold; p_out = false)
    
    exposed = data.Exposed
    alive = data.Alive
    times = data.Time
    
    exposed_stats = t_test(exposed)
    alive_stats = t_test(alive)
    time_stats = t_test(times)
    
    #These results are unfiltered and contain runs where ASF dies off straight away
    mask = exposed .> threshold
    
    exp_f = exposed[mask]
    alive_f = alive[mask]
    time_f = times[mask]
    
    
    exp_f_stats = t_test(exp_f)
    alive_f_stats = t_test(alive_f)
    time_f_stats = t_test(time_f)
    
    if p_out == true
        println("Exposed!")
        println("Mean- $(exposed_stats[2]) & $(exposed_stats[1]) - $(exposed_stats[3])")
        println()
        println("Alive!")
        println("Mean- $(alive_stats[2]) & $(alive_stats[1]) - $(alive_stats[3])")
        println()
        println("Time!")
        println("Mean- $(time_stats[2]) & $(time_stats[1]) - $(time_stats[3])")
        println()
        println("Filtered")
        println()
        println("Exposed!")
        println("Mean- $(exp_f_stats[2]) & $(exp_f_stats[1]) - $(exp_f_stats[3])")
        println()
        println("Alive!")
        println("Mean- $(alive_f_stats[2]) & $(alive_f_stats[1]) - $(alive_f_stats[3])")
        println()
        println("Time!")
        println("Mean- $(time_f_stats[2]) & $(time_f_stats[1]) - $(time_f_stats[3])")
    end
    
    return exposed_stats, alive_stats, exp_f_stats, alive_f_stats, time_stats, time_f_stats
    
end