"Module that fills the transmission matrix from either fitted parameters of specified input"

module Beta

using CSV
using DataFrames

export construction

function construction(sim, pops, counts, network)
    #this function is to convert the base network we have created into a transmission coefficant matrix


    n_p = sim.N_Pop # number of populations per region
    n_r = size(pops)[1] #number of regions
    n_pops = n_p*n_r #total number of populations
   
    n_cs = counts.cum_sum

    beta = Float32.(copy(network))
    connected_births = copy(network)
    
    connected_births[connected_births .!= 200] .= 0 #only wanted connected groups within same pop
    connected_pops =  connected_births .÷ 200 #setting to ones
    
    beta_inter = zeros(Float32,n_pops)

    Mn = sim.Model
    
    if Mn == 3 #building beta on network!
        for i in 1:n_pops #iterating through

            j = (i-1) ÷ n_p + 1

            data = pops[j]
            #connected_pops[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]] *= i
            beta_pop = beta[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]]
            

            if !sim.Fitted
                if sim.Identical #no variation, just mean of dist

                    beta_pop[beta_pop .== 100] .= data.B_f[1] #intra feral
                    beta_pop[beta_pop .== 200] .= data.B_ff[1] #inter feral
                    beta_pop[beta_pop .== 300] .= data.B_fl[1] #farm feral
                    beta_pop[beta_pop .== 400] .= data.B_l[1] #intra farm
                    beta_inter[i] = data.B_ff[1]
                else #from dist
                    i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
                    i_ff = TruncatedNormal(data.B_ff[1],data.B_ff[2],0,5) #inter group

                    n_intra = length(beta_pop[beta_pop .== 100])
                    n_inter = length(beta_pop[beta_pop .== 200])

                    b_intra = rand(i_f, n_intra)
                    b_inter = rand(i_ff, n_inter) 

                    beta_pop[beta_pop .== 100] = b_intra
                    beta_pop[beta_pop .== 200] = b_inter  

                    beta_inter[i] = rand(i_ff) 

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
                    
                    beta_pop = beta_pop

                end
            else #running M3 from fitted values!
                if sim.Network == "s"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Scale_Free/"
                elseif sim.Network == "w"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Small_Worlds/"
                else 
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Random/"
                end

                rand_values = rand(1:10000,2)

                df_beta_intra = Array(CSV.read(path*"/contact_in.csv", DataFrame, header=false))
                df_beta_inter = Array(CSV.read(path*"/contact_out.csv", DataFrame, header=false))

                beta_intra  = df_beta_intra[rand_values[1]][1]
                beta_inter = df_beta_inter[rand_values[2]][1]

                beta_pop[beta_pop .== 100] .= beta_intra #intra feral
                beta_pop[beta_pop .== 200] .= beta_inter/6 #inter feral
                beta_pop[beta_pop .== 300] .= data.B_fl[1] #farm feral not_fitted!
                beta_pop[beta_pop .== 400] .= data.B_l[1] #intra farm not fitted!
                
                beta_inter[i] = beta_inter/6
            end

            beta[n_cs[i]+1:n_cs[i+1],n_cs[i]+1:n_cs[i+1]] = beta_pop

        end
    elseif Mn == 2 #tau Homogeneous model
        data = pops[1]
        if !sim.Fitted
            if sim.Identical #no variation, just mean of dist

                beta[1] = data.B_f[1] #intra feral
                
            else #from dist
                i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
                beta[1] = rand(i_f, 1)
            end
        else #running M3 from fitted values!
            path = "Inputs/Fitted_Params/Tau-Homogeneous/contact_out.csv"
            rand_values = rand(1:10000,1)

            df_beta = Array(CSV.read(path, DataFrame, header=false))
            beta_intra  = df_beta[rand_values][1]
            
            beta[1] = beta_intra
        end
    else #ODE MODEL
        data = pops[1]
        if !sim.Fitted
            if sim.Identical #no variation, just mean of dist

                beta[1] = data.B_f[1] #intra feral
                
            else #from dist
                i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
                beta[1] = rand(i_f, 1)
            end
        else #running M3 from fitted values!
            path = "Inputs/Fitted_Params/ODE/contact_out.csv"
            rand_values = rand(1:10000,1)

            df_beta = Array(CSV.read(path, DataFrame, header=false))
            beta_intra  = df_beta[rand_values][1]
            beta[1] = beta_intra
        end
    end 

    return beta, connected_pops, beta_inter

end

end
