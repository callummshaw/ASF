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

function Model(input_path,out)
    #wrapper function to run ASF models!

    i1 = Input.Model_Data(input_path, verbose = true)
   
    n_sims  = i1.NR
    n_time  = i1.Time
    n_group = i1.Parameters.Populations.total[1]

    MN = i1.MN
    if out == "f"
        @info "Full simulation output"
        output = zeros(n_sims,round(Int,n_time[2]-n_time[1])+1,n_group*5) 
    else
        @info "Summary statistic output"
        if MN in [1,2]
            output = zeros(n_sims,8) #8 summary stats
        else
            output = zeros(n_sims,9)
        end
    end

    for i in 1:i1.NR
        
        
        input = Input.Model_Data(input_path); #all input data!
    
        if MN == 1 #running ODE model!

            params = convert_homogeneous(input.Parameters) #converting params to single array!

            prob_ode = ODEProblem(Models.ASF_M1, input.U0, input.Time, params) #setting up ode model
            
            sol = solve(prob_ode, saveat = 1,reltol=1e-8) #running ode model!
            
        else #running TAU model!
           
            nt = input.Parameters.Populations.cum_sum[end] #total number of groups and/or farms
            
            nc = 5 #number of classes (SEIRC)
            eqs = 11 #number of processes

            if nt > 5000
                @warn "Transition matrix too large, reduce group numbers"
            end
            #Matrix of all the transitions between classes for Gillespie model
            dc = sparse(zeros(Float32,nt*nc,nt*eqs))

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

            if MN == 2 #M2!

                params = convert_homogeneous(input.Parameters) #converting params to single array!
               
                rj = RegularJump(Models.ASF_M2, regular_c, eqs)
                prob = DiscreteProblem(input.U0,input.Time, params)
                jump_prob = JumpProblem(prob,Direct(),rj)
                sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
                
            elseif (MN == 3) & (input.Parameters.Populations.pop == 1) #M3 with one pop!
                
                
                params = convert_heterogeneous(input.Parameters) #converting params to single array!
                
                rj = RegularJump(Models.ASF_M3_single, regular_c, eqs*nt)
                U0 = convert(Vector{Int8}, input.U0)
                prob = DiscreteProblem(U0,input.Time, params)#hetero_single_test(input.Parameters))
                jump_prob = JumpProblem(prob,Direct(),rj)
                sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
                
            else #M3 with multipopulations!
            
                rj = RegularJump(Models.ASF_M3_full, regular_c, eqs*nt)
                U0 = convert(Vector{Int16}, input.U0)
                
                prob = DiscreteProblem(U0,input.Time, input.Parameters)
                jump_prob = JumpProblem(prob,Direct(),rj)
                sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

            end

        end

        if out == "f"
            data = reduce(vcat,transpose.(sol.u))

            if any(x->x==-1, data) == true
                println("Need to Reduce Timestep")
            end

            output[i,:,:] = data  

        else
            
            if MN in [1,2]
                output[i,:] = Analysis.summary_stats_homogeneous(sol, false)
            else 
                output[i,:] = Analysis.summary_stats_heterogeneous(sol,false)
            end

        end

    end
    return output
end

function Model_sim(input_path, bm,dm)
    #wrapper function to run ASF models!

    input = Input.Model_Data(input_path, bm,dm, verbose = false); #all input data!

    n_sims  = input.NR
    n_time  = input.Time
    n_group = input.Parameters.Populations.total[1]

    MN = input.MN
   
    if MN == 1 #running ODE model!

        params = convert_homogeneous(input.Parameters) #converting params to single array!
        
        prob_ode = ODEProblem(Models.ASF_M1, input.U0, input.Time, params) #setting up ode model
        
        sol = solve(prob_ode, saveat = 1,reltol=1e-8) #running ode model!
        
    else #running TAU model!
         
        nt = input.Parameters.Populations.cum_sum[end] #total number of groups and/or farms
        
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

        if MN == 2 #M2!

            params = convert_homogeneous(input.Parameters) #converting params to single array!
            
            rj = RegularJump(Models.ASF_M2, regular_c, eqs)
            prob = DiscreteProblem(input.U0,input.Time, params)
            jump_prob = JumpProblem(prob,Direct(),rj)
            sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
            
        elseif (MN == 3) & (input.Parameters.Populations.pop == 1) #M3 with one pop!
           #=
            U = Vector{Vector{Int8}}(undef,n_sims)
            P = Vector{Vector{Any}}(undef,n_sims)
        
                inputs = Input.Model_Data(input_path, verbose = false)
                u = convert(Vector{Int8}, inputs.U0)
                p = convert_heterogeneous(inputs.Parameters)
                U[i] = u
                P[i] = p 
              
           =#
            function prob_func(prob, i, repeat)
                
                input_new =  Input.Model_Data(input_path, bm, dm, verbose = false)
                un = convert(Vector{Int8}, input_new.U0)
                pn = convert_heterogeneous(input_new.Parameters)
                
                remake(prob, u0 = un, p = pn)
            end

            function output_func(sol,i) 
                GC.gc()
                (sum(sol[end][2:5:end] + sol[end][3:5:end] + sol[end][5:5:end]), false)
                
            end
            
            rj = RegularJump(Models.ASF_M3_single, regular_c, eqs*nt)
            
            U = convert(Vector{Int8}, input.U0)
            P = convert_heterogeneous(input.Parameters)
            prob = DiscreteProblem(U,input.Time, P)#hetero_single_test(input.Parameters))
            jump_prob = JumpProblem(prob,Direct(),rj)

            ensemble_prob = EnsembleProblem(jump_prob, prob_func = prob_func, output_func = output_func)
            sim = solve(ensemble_prob, SimpleTauLeaping(),  EnsembleThreads(), dt =1, trajectories = n_sims)

            #sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

        else #M3 with multipopulations!
        
            rj = RegularJump(Models.ASF_M3_full, regular_c, eqs*nt)
            U0 = convert(Vector{Int16}, input.U0)
            
            prob = DiscreteProblem(U0,input.Time, input.Parameters)
            jump_prob = JumpProblem(prob,Direct(),rj)
            sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

        end

    end

    #end_out = Analysis.summary_stats_endemic(sol)
    #data_wanted = sum(sol[end][2:5:end] + sol[end][3:5:end] +sol[end][5:5:end])
    #output[i] = end_out 
    


    return sim
end

function birthpulse_norm(s, DT)
   
    integral, err = quadgk(x -> exp(-s*cos(pi*x/365)^2), 1, 365, rtol=1e-8);
    k = (365*DT)/integral 
    
    return k
end

function convert_homogeneous(input)
    #Function to convert input structure to simple array for M1 and M2
    params = Vector{Any}(undef,18)

    params[1]  = input.β_o[1][1]
    params[2]  = input.μ_p[1][1]
    params[3]  = input.K[1][1]
    params[4]  = input.ζ[1][1]
    params[5]  = input.γ[1][1]
    params[6]  = input.ω[1][1]
    params[7]  = input.ρ[1][1]
    params[8]  = input.λ[1][1]
    params[9]  = input.κ[1][1]
    
    params[10] = input.σ[1]
    params[11] = input.θ[1]
    params[12] = input.Seasonal
    params[13] = input.bw[1]
    params[14] = input.bo[1]
    params[15] = input.k[1]
    params[16] = input.la[1]
    params[17] = input.lo[1]
    params[18] = input.area[1]
    
    return params
end
    
function convert_heterogeneous(input)
    
    params =  Vector{Any}(undef,22)
    
    params[1]  = input.β_o[1] #inter
    params[2]  = input.β_i[1] #intra
    
    params[3]  = input.μ_p[1]
    params[4]  = input.K[1]
    params[5]  = input.ζ[1]
    params[6]  = input.γ[1]
    params[7]  = input.ω[1]
    params[8] = input.ρ[1]
    params[9] = input.λ[1]
    params[10] = input.κ[1]
    
    params[11] = input.σ[1]
    params[12] = input.g[1]

    params[13] = input.bw[1]
    params[14] = input.bo[1]
    params[15] = input.k[1]
    params[16] = input.la[1]
    params[17] = input.lo[1]
    
    params[18] = input.area[1]
    params[19] = input.Populations.inter_connections[1]
    params[20] = input.Populations.networks[1]
    
    params[21] = ones(Int16, length(input.K[1]))
    params[22] = ones(Int16, length(input.K[1]))
            
    return params
end

end  