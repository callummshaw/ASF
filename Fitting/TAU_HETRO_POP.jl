module SIR_TAU_Pop

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Distributions
using Random

include("/home/callum/ASF/Modules/ASF_input.jl");


#Tspan params
start_day = 0
n_years = 5
Tspan = (start_day,n_years*365+start_day)


input_path = "/home/callum/ASF/Inputs/"; #path to model data
input = ASF_Inputs.Model_Data(input_path)

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

function distance(Y,Y0)
    #Distance function for ABC fitting
    U0 = Y0["SS"] 
    U = Y["SS"]

    vals = (U0 .- U)
    
    d = sqrt(sum(vals.^2))
    
    return d
end

function birth_pulse_vector(t,k,s,p)
    #birth pulse not for a vector
    return k*exp(-s*cos(pi*(t+p)/365)^2)
end


function model_1(par)
    
    p1 = par["p1"]
    input_path = "/home/callum/ASF/Inputs/"; #path to model data

    input = ASF_Inputs.Model_Data(input_path)
    
    params = convert(input.Parameters)
    
    U0 = input.U0
    U0[1:5:end] += (U0[2:5:end] +U0[3:5:end]) #setting init pop to zero
    U0[2:5:end] .= 0 #removing infected populations
    U0[3:5:end] .= 0
    
    

    #beta
    params[15] = p1 #intra
   
    rj = RegularJump(asf_model_pop, regular_c, eqs*nt)
    prob = DiscreteProblem(U0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = sum(hcat(sol.u...), dims=1)
    
    return Dict("SS"=>summary)
    
end

function model_2(par)
    
    p1 = par["p1"]
    input_path = "/home/callum/ASF/Inputs_sf/"; #path to model data

    input = ASF_Inputs.Model_Data(input_path)
    
    params = convert(input.Parameters)
    
    U0 = input.U0
    U0[1:5:end] += (U0[2:5:end] +U0[3:5:end]) #setting init pop to zero
    U0[2:5:end] .= 0 #removing infected populations
    U0[3:5:end] .= 0
    
    

    #beta
    params[15] = p1 #intra
   
    rj = RegularJump(asf_model_pop, regular_c, eqs*nt)
    prob = DiscreteProblem(U0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = sum(hcat(sol.u...), dims=1)
    
    return Dict("SS"=>summary)
    
end


function model_3(par)
    
    p1 = par["p1"]
    input_path = "/home/callum/ASF/Inputs_r/"; #path to model data

    input = ASF_Inputs.Model_Data(input_path)
    
    params = convert(input.Parameters)
    
    U0 = input.U0
    U0[1:5:end] += (U0[2:5:end] +U0[3:5:end]) #setting init pop to zero
    U0[2:5:end] .= 0 #removing infected populations
    U0[3:5:end] .= 0
    
    

    #beta
    params[15] = p1 #intra
   
    rj = RegularJump(asf_model_pop, regular_c, eqs*nt)
    prob = DiscreteProblem(U0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = sum(hcat(sol.u...), dims=1)
    
    return Dict("SS"=>summary)
    
end



function convert(input)
    #Function to convert input structure to simple array (used mainly for fitting)
    params = Vector{Any}(undef,22)
    
    beta = copy(input.β)
    beta_con = copy(input.β_b)

    params[1]  = beta[diagind(beta)]
    params[2]  = beta.*beta_con
    params[3]  = copy(input.β_d)
    
    params[4]  = copy(input.μ_p)
    params[5]  = copy(input.K)
    params[6]  = copy(input.ζ[1])
    params[7]  = copy(input.γ[1])
    params[8]  = copy(input.ω[1])
    params[9] = copy(input.ρ[1])
    params[10] = copy(input.λ[1])
    params[11] = copy(input.κ[1])
    
    params[12] = copy(input.σ[1])
    params[13] = copy(input.θ[1])
    params[14] = copy(input.η[1])
    params[15] = copy(input.g[1])

    params[16] = copy(input.Seasonal)
    params[17] = copy(input.bw[1])
    params[18] = copy(input.bo[1])
    params[19] = copy(input.k[1])
    params[20] = copy(input.la[1])
    params[21] = copy(input.lo[1])
    
    params[22] = copy(input.Populations.area[1])
    return params
end

function asf_model_pop(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β_i, β_o, β_b, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, Seasonal, bw, bo, k, la, lo, Area  = p 
    
    
    year = 365 #days in a year

    u[u.<0] .= 0
    
    S = Vector{UInt32}(u[1:5:end])
    
    tg = length(S)
    
    p_mag = birth_pulse_vector(t,k,bw,0)
    Births = p_mag.*(σ .* S .+ ((1-σ)) .* sqrt.(S .* K))#S.^(1-θ) .* K.^θ)
    
    #now stopping boar births
    mask_boar = (K .== 1) .& (S .> 0) #boars with a positive population
    boar_births = p_mag*sum(mask_boar)
    Births[mask_boar] .= 0
    mask_p_s = (S .> 1) .& (K .> 1) #moving it to postive 
    Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
     
    
    n_empty  = sum(S .== 0 ) 
    n_r = (n_empty/tg)^2
    
    
    if (n_empty/tg) > 0.01
        
        dd = copy(S)
        dd[dd .< 2] .= 0
        connected_pops = β_b * dd

            #Groups with 3 or more pigs can have emigration
        mask_em =  (dd .> 0) #populations that will have emigration

        em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

        mask_im = (S .== 0) .& (connected_pops .> 1) #population zero but connected groups have 5 or more pigs

        Births[mask_em] .*= n_r
        Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)
    end 
    out[1:11:end] .= Births
    out[2:11:end] .= S.*μ_p.*(σ .+ (1-σ).*sqrt.(S./K))*g
   
    nothing
end

function density_carrying!(du,u,p,t)
    #ODE to calculate the "real" observations
    S = u[1]
    k, bw, σ, K, μ_p = p
  
    du[1] =  k*exp(-bw*cos(pi*(t)/365)^2).*(σ .* S .+ ((1-σ)) .* sqrt.(S.*K))-S.*μ_p.*(σ .+ ((1-σ)).*sqrt.(S./K))
    
end

function observation()
   U0_ode = [sum(input.U0)];
   p_ode = [input.Parameters.k[1], input.Parameters.bw[1],input.Parameters.σ[1],sum(input.Parameters.K),input.Parameters.μ_p[1]];
   
    #solving ODE
    prob_ode = ODEProblem(density_carrying!, U0_ode, (0.0,n_years*365.0), p_ode)
    sol_ode = solve(prob_ode, saveat = 1,reltol=1e-8)
    obs = hcat(sol_ode.u...); 
    
    return Dict("SS"=>obs)
    
end



end
