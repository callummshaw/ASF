"Module that fills the transmission matrix from either fitted parameters of specified input"

module Beta

using CSV
using DataFrames
using Random

export construction

function construction(sim, pops, counts)
    
    n_p = sim.N_Pop # number of populations per region
    n_r = size(pops)[1] #number of regions
    n_pops = n_p*n_r #total number of populations
    Mn = sim.Model

    beta_o = Vector{Vector{Float32}}(undef,n_pops) #inter_beta
    beta_i = Vector{Vector{Float32}}(undef,n_pops) #intra_beta
        
    if Mn == 3 #building beta on network!
        for i in 1:n_pops #iterating through

            j = (i-1) ÷ n_p + 1
            data = pops[j]
        
            ng = counts.feral[i]

            if !sim.Fitted
               
                i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
                i_ff = TruncatedNormal(data.B_ff[1],data.B_ff[2],0,5) #inter group

                b_intra = rand(i_f, ng)
                b_inter = rand(i_ff, ng) 

                beta_i[i] = b_intra
                beta_o[i] = b_inter

            else #running M3 from fitted values!
                if sim.Network == "s"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Scale_Free/"
                elseif sim.Network == "w"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Small_Worlds/"
                else 
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Random/"
                end

                df_beta_intra = shuffle(Array(CSV.read(path*"/contact_in.csv", DataFrame, header=false)))
                df_beta_inter = shuffle(Array(CSV.read(path*"/contact_out.csv", DataFrame, header=false)))

                beta_i[i] = df_beta_intra[1:ng]
                beta_o[i] = df_beta_inter[1:ng]./6
                
            end


        end
    elseif Mn == 2 #tau Homogeneous model
        
        data = pops[1]
        if !sim.Fitted
            
                i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
                beta_o[1] = [rand(i_f, 1)]

        else #running M2 from fitted values!
            path = "Inputs/Fitted_Params/Tau-Homogeneous/contact_out.csv"

            df_beta = shuffle(Array(CSV.read(path, DataFrame, header=false)))
            
            beta_o[1] = [df_beta[1]]
        end
    else #ODE MODEL
        data = pops[1]
        if !sim.Fitted
           #from dist
            i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
            beta_o[1] = [rand(i_f, 1)]
        
        else #running M2 from fitted values!
            path = "Inputs/Fitted_Params/ODE/contact_out.csv"

            df_beta = shuffle(Array(CSV.read(path, DataFrame, header=false)))

            beta_o[1] = [df_beta[1]]
        end
    end 

    return beta_o, beta_i

end

end
