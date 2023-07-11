"Wrapper module to run ASF modelling framework"
module ASF

export Model
export Model_sim

"""
   Model(input_path,out)
Function that runs the ASF models, ``input_path`` is the location of input data, which are three csv files. ``Simulation_Data.csv`` contains model meta data, ``Population.csv`` contains data on the population being modelled, and ``Seasonal.csv`` contains information for seasonally varying parameters. Descriptions of each variable specified in the csv files are included in the sample input files provided. The second input ``out`` controls the model output type. By default it is set to return summary statstics of the runs, however if ``out`` is set to ``f`` full model simualtions on a daily time-step are returned 
"""


using DifferentialEquations

using QuadGK
using LinearAlgebra
using Distributions
using SparseArrays

include("Models.jl") #where all the models are!
include("Input.jl") #the input
include("Analyse.jl") #some simple analysis


function Model_sim(input_path; pop_net = 0, year_array = 0, fym = 0.95)
    #wrapper function to run ASF models!

    input = Input.Model_Data(input_path, pop_net, year_array, fym, verbose = false); #all input data!

    n_sims  = input.NR
    n_pops = input.Parameters.Populations.pop
    groups_per_pop = input.Parameters.Populations.cum_sum
    MN = input.MN

    if MN == 1 #running ODE model!

        params = convert_homogeneous(input.Parameters) #converting params to single array!
        
        prob_ode = ODEProblem(Models.ASF_M1, input.U0, input.Time, params) #setting up ode model
        
        sim = solve(prob_ode, saveat = 1,reltol=1e-8) #running ode model!
        
    else #running TAU model!
         
        nt = groups_per_pop[end] #total number of groups and/or farms
        
        nc = 5 #number of classes (SEIRC)
        eqs = 11 #number of processes

        #Matrix of all the transitions between classes for Gillespie model
        dc = sparse(zeros(nt*nc,nt*eqs))

        dc[0*nc*nt+1:nc*nt*eqs+nc:end] .= 1
        dc[1*nc*nt+1:nc*nt*eqs+nc:end] .= -1
        dc[2*nc*nt+1:nc*nt*eqs+nc:end] .= -1
        dc[10*nc*nt+1:nc*nt*eqs+nc:end] .= 1

        dc[2*nc*nt+2:nc*nt*eqs+nc:end] .= 1
        dc[3*nc*nt+2:nc*nt*eqs+nc:end] .= -1
        dc[4*nc*nt+2:nc*nt*eqs+nc:end] .= -1

        dc[4*nc*nt+3:nc*nt*eqs+nc:end] .= 1
        dc[5*nc*nt+3:nc*nt*eqs+nc:end] .= -1
        dc[6*nc*nt+3:nc*nt*eqs+nc:end] .= -1
        dc[7*nc*nt+3:nc*nt*eqs+nc:end] .= -1

        dc[7*nc*nt+4:nc*nt*eqs+nc:end] .= 1
        dc[8*nc*nt+4:nc*nt*eqs+nc:end] .= -1
        dc[10*nc*nt+4:nc*nt*eqs+nc:end] .= -1


        dc[5*nc*nt+5:nc*nt*eqs+nc:end] .= 1
        dc[6*nc*nt+5:nc*nt*eqs+nc:end] .= 1
        dc[9*nc*nt+5:nc*nt*eqs+nc:end] .= -1;

        function regular_c(du,u,p,t,counts,mark)  
            mul!(du,dc,counts)
            nothing
        end

        function output_func(sol,i) 
           
            GC.gc()

            data = reduce(vcat,transpose.(sol.u))
            
            dd = Vector{Vector{Int64}}(undef,n_pops)
            fd = Vector{Vector{Int64}}(undef,n_pops)
            for i in 1:n_pops

                s0 = groups_per_pop[i] #start index of pop
                ei = groups_per_pop[i+1] #end index of pop
                
                s_d = sum(data[:,5*s0+1:5:5*ei],dims=2)
                e_d = sum(data[:,5*s0+2:5:5*ei],dims=2)
                i_d = sum(data[:,5*s0+3:5:5*ei],dims=2)
                r_d = sum(data[:,5*s0+4:5:5*ei],dims=2)
                c_d = sum(data[:,5*s0+5:5:5*ei],dims=2)
   
                dis = vec(e_d +i_d + c_d)
                free_dis = vec(s_d + r_d)
                
                dd[i] = dis[1:7:end] #weekly output
                fd[i] = free_dis[1:7:end]
            end
            
            ([dd,fd], false)
            
        end

        function prob_func(prob, i, repeat)
            input_new =  Input.Model_Data(input_path, pop_net, year_array, fym, verbose = false)
            
            if MN == 2
                pn = convert_homogeneous(input_new.Parameters)
            else 
                if input_new.Parameters.Populations.pop == 1 
                    pn = convert_heterogeneous(input_new.Parameters)
                else 
                    pn = input_new.Parameters
                end
            end

            remake(prob, u0 = input_new.U0, p = pn)
        end

        if MN == 2 #M2!

            rj = RegularJump(Models.ASF_M2, regular_c, eqs)
                
            P = convert_homogeneous(input.Parameters) #convert to a simple vector of inputs as most not needed
            
        elseif (MN == 3) 
        
            if input.Parameters.Populations.pop == 1 #M3 with one pop!
                
                rj = RegularJump(Models.ASF_M3_single, regular_c, eqs*nt)
                
                P = convert_heterogeneous(input.Parameters) #convert to a simple vector of inputs as most not needed
            
            else #running multi!
                rj = RegularJump(Models.ASF_M3_full, regular_c, eqs*nt)

                P = input.Parameters
            
            end

        end
        
        prob = DiscreteProblem(input.U0,input.Time, P)#hetero_single_test(input.Parameters))
        jump_prob = JumpProblem(prob,Direct(),rj)

        ensemble_prob = EnsembleProblem(jump_prob, prob_func = prob_func, output_func = output_func)
        sim = solve(ensemble_prob, SimpleTauLeaping(),  EnsembleThreads(), dt =1, trajectories = n_sims)

    end

    return sim
end

function birthpulse_norm(s, DT)
   
    integral, err = quadgk(x -> exp(-s*cos(pi*x/365)^2), 1, 365, rtol=1e-8);
    k = (365*DT)/integral 
    
    return k
end

function convert_homogeneous(input)
    #Function to convert input structure to simple array for M1 and M2
    

    params = (input.β_o[1][1],input.μ_p[1],input.K[1][1],input.ζ[1][1],input.γ[1][1],input.ω[1][1],input.ρ[1][1],input.λ[1][1],input.κ[1][1],input.σ[1],input.θ[1],input.Seasonal,input.bw[1],input.bo[1],input.k[1],input.la[1],input.lo[1],input.area[1])
    
    return params
end
    
function convert_heterogeneous(input)
   
    params = (input.β_o[1], input.β_i[1],input.μ_p[1],input.K[1],input.ζ[1],input.γ[1],input.ω[1],input.ρ[1],input.λ[1],input.κ[1],input.σ[1],input.g[1],input.bw[1],input.bo[1],input.k[1],input.la[1],input.lo[1],input.area[1],input.Populations.inter_connections[1],input.Populations.networks[1],input.ds1, input.ds2)
    
    return params
end

end  
