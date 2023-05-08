module Run
#module that runs the African swine fever models, a wrapper for all other modules

include("Models.jl") #where all the models are!
include("Input.jl") #the input
include("Analyse.jl")

function ASF(input_path, out)
    #wrapper function to run ASF models!

    i1 = Input.Model_Data(input_path)

    n_sims  = i1.NR
    n_time  = i1.Time
    n_group = i1.Populations.total
    MN = i1.MN
    if out == "f"
        @info "Full simulation output"
        output = zeros(n_sims,n_time,n_group*5) 
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

            prob_ode = ODEProblem(Models.ASF_M1, Input.U0, Input.Time ,params) #setting up ode model
            
            sol = solve(prob_ode, saveat = 1,reltol=1e-8) #running ode model!

        else #running TAU model!

            const nt = input.Parameters.Populations.cum_sum[end] #total number of groups and/or farms
            const nc = 5 #number of classes (SEIRC)
            const eqs = 11 #number of processes

            #Matrix of all the transitions between classes for Gillespie model
            const dc = sparse(zeros(nt*nc,nt*eqs))

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
                prob = DiscreteProblem(Input.U0,Input.Time, params)
                jump_prob = JumpProblem(prob,Direct(),rj)
                sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

            elseif (MN == 3) & (input.Parameters.Populations.pop == 1) #M3 with one pop!

                params = convert_heterogeneous(input.Parameters) #converting params to single array!
                rj = RegularJump(Models.ASF_M3S, regular_c, eqs)
                prob = DiscreteProblem(Input.U0,Input.Time, params)
                jump_prob = JumpProblem(prob,Direct(),rj)
                sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

            else #M3 with multipopulations!

                rj = RegularJump(Models.ASF_M3, regular_c, eqs)
                prob = DiscreteProblem(Input.U0,Input.Time, Input.Parameters)
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
                output[i,:] = Analysis.summary_stats_homogeneous(sol)
            else 
                output[i,:] = Analysis.summary_stats_heterogeneous(sol)
            end

        end

    end

end


function convert_homogeneous(input)
    #Function to convert input structure to simple array for M1 and M2
    params = Vector{Any}(undef,19)

    params[1]  = input.β[1]
    params[2]  = input.μ_p[1]
    params[3]  = input.K[1]
    params[4]  = input.ζ[1]
    params[5]  = input.γ[1]
    params[6]  = input.ω[1]
    parmsams[7] =input.ρ[1]
    params[8] = input.λ[1]
    params[9] = input.κ[1]
    
    params[10] = input.σ[1]
    params[11] = input.θ[1]
    params[12] = input.η[1]
    params[13] = input.Seasonal
    params[14] = input.bw[1]
    params[15] = input.bo[1]
    params[16] = input.k[1]
    params[17] = input.la[1]
    params[18] = input.lo[1]
    params[19] = input.Populations.area[1]
    
    return params
end
    
  


function convert_heterogeneous(input)
    #Function to convert input structure to simple array, only for single population M3 model!
    params = Vector{Any}(undef,22)
    
    beta = input.β
    beta_con = input.β_b

    params[1]  = beta[diagind(beta)]
    params[2]  = beta.*beta_con
    params[3]  = input.β_d
    
    params[4]  = input.μ_p[1]
    params[5]  = input.K
    params[6]  = input.ζ[1]
    params[7]  = input.γ[1]
    params[8]  = input.ω[1]
    params[9] = input.ρ[1]
    params[10] = input.λ[1]
    params[11] = input.κ[1]
    
    params[12] = input.σ[1]
    params[13] = input.θ[1]
    params[14] = input.η[1]
    params[15] = input.g[1]

    params[16] = input.Seasonal
    params[17] = input.bw[1]
    params[18] = input.bo[1]
    params[19] = input.k[1]
    params[20] = input.la[1]
    params[21] = input.lo[1]
    
    params[22] = input.Populations.area[1]

    return params
end



end