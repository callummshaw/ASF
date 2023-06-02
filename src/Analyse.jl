"Module to analyse ASF model output and create relevant summary statistics"
module Analysis

using Statistics

export summary_stats_homogeneous
export summary_stats_heterogeneous
export summary_stats_endemic

function summary_stats_homogeneous(sol,verbose)
    #=
    1. Peak time (time of infection peak)
    2. Endemic (1 for endemic, 0 for dieout)
    3. Endemic count live inf (0 if non-endemic )
    4. Endemic count corpses (0 if non-endemic )
    5. Endemic count alive (0 if non-endemic)
    6. Die out time (0 for endemic otherwise time at dieout)
    7. Population min 
    8. Infection max
    =#
    stats = 8
    
    output = Array{Float64}(undef, stats)

    data = reduce(vcat,transpose.(sol.u))

    if any(x->x <0, data)
        println("Need to Reduce Timestep")
        data[data .< 0 ] .= 0
    end
    
    s_d = data[:,1]
    e_d = data[:,2]
    i_d = data[:,3]
    r_d = data[:,4]
    c_d = data[:,5]

    total_time  = size(s_d)[1]
    t_end = 180
    disease = e_d + i_d + c_d #classes with disease

    max_d = findmax(disease) #max time
    max_dt = max_d[2][1]
    
    output[1] = max_dt
    output[8] = max_d[1][1]
    if disease[end] >= 1 #more than 1 infected pig/ corpse    
        
        output[2] = 1
        output[6] = 0
      
        if total_time - (max_dt + t_end) < 365
            if verbose
            @warn "Endemic but peak in infections too close to end of simulation, increase time!"
            end
            output[3] = -1
            output[4] = -1
            output[5] = -1
        else
            output[3] = mean(e_d[max_dt+t_end : end] + i_d[max_dt+t_end : end])
            output[4] = mean(c_d[max_dt+t_end : end])
            output[5] = mean(e_d[max_dt+t_end : end] + i_d[max_dt+t_end : end]+s_d[max_dt+t_end : end] + r_d[max_dt+t_end : end])
        end
    else 
        output[2] = 0
        output[3] = 0
        output[4] = 0
        output[5] = 0
        output[6] = findfirst(disease .< 1)[1] #non endemic!
    end

    output[7] = findmin(s_d+e_d+i_d+r_d)[1]
    

    return output
end

function summary_stats_endemic(sol)
    

    data = reduce(vcat,transpose.(sol.u))

    
    s_d = data[:,1]
    e_d = data[:,2]
    i_d = data[:,3]
    r_d = data[:,4]
    c_d = data[:,5]

   
    disease = e_d + i_d + c_d #classes with disease

    if disease[end] >= 1 #more than 1 infected pig/ corpse    
        
        output = 9999
    else
        output = findfirst(disease .< 1)[1] #non endemic!
    end
    

    return output
end

function summary_stats_heterogeneous(sol,verbose)
    #=
    1. Peak time (time of infection peak)
    2. Endemic (1 for endemic, 0 for dieout)
    3. Endemic count live inf (0 if non-endemic )
    4. Endemic count corpses (0 if non-endemic )
    5. Endemic count alive (0 if non-endemic)
    6. Die out time (0 for endemic otherwise time at dieout)
    7. Maximum number of dead groups at one time
    8. Maximum number of infected groups at one time
    9. Total groups infected
    =#
    stats = 9
    
    output = Array{Float64}(undef, stats)

    data = reduce(vcat,transpose.(sol.u))

    if any(x->x <0, data)
        if sum(data .< 0) > 100
            @warn "Reduce timestep"
        end
         
        data[data .< 0 ] .= 0
    end
    
    s_d = data[:,1:5:end]
    e_d = data[:,2:5:end]
    i_d = data[:,3:5:end]
    r_d = data[:,4:5:end]
    c_d = data[:,5:5:end]

    total_time  = size(s_d)[1]
    t_end = 180

    disease_total = e_d + i_d + c_d #classes with disease,
    disease_alive = e_d + i_d
 
    disease_free = s_d + r_d #classes without disease,
 
    disease_sum = sum(disease_total,dims=2)
    disease_alive_sum =  sum(disease_alive,dims=2)
    disease_corpse_sum = sum(c_d, dims = 2)
    disease_free_sum = sum(disease_free,dims=2)
    population_sum = disease_alive_sum + disease_free_sum;
   
    max_d = findmax(disease_sum) #max time
    max_dt = max_d[2][1]

    output[1] = max_dt

    if disease_sum[end] >= 1 #more than 1 infected pig/ corpse    
        
        output[2] = 1
        output[6] = 0

        if total_time - (max_dt + t_end) < 365
            if verbose
            @warn "Endemic but peak in infections too close to end of simulation, increase time!"
            end
            output[3] = -1
            output[4] = -1
            output[5] = -1
        else
            output[3] = mean(disease_alive_sum[max_dt+t_end : end])
            output[4] = mean(disease_corpse_sum[max_dt+t_end : end])
            output[5] = mean(population_sum[max_dt+t_end : end])
        end
    else 
        output[2] = 0
        output[3] = 0
        output[4] = 0
        output[5] = 0
        output[6] = findfirst(disease_sum .< 1)[1] #non endemic!
    end

    pop = disease_alive + disease_free
    dead_groups = sum(pop .==0 ,dims =2)
    
    output[7] = findmax(dead_groups)[2][1]

    inf_groups = sum(disease_total .> 0 , dims = 2) 
    output[8] = findmax(inf_groups)[2][1] #maximum  number of groups infected at one time

    one_inf_groups = sum(disease_total, dims =1)
    output[9] = sum(one_inf_groups .> 0) #total number of groups infected

    return output
end


end
