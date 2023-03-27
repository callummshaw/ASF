module SIR_TAU_S

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Distributions

include("/home/callum/ASF/Modules/ASF_input.jl");


#Tspan params
start_day = 180.0
n_years = 3
Tspan = (start_day,n_years*365+start_day)
#seeding init pop
n_exp = 30 #number of exposed
n_inf = 20 #number of infected
N_total = 6518 # total population at start date (180) found from previous burn in
u0 = [N_total-n_exp-n_inf,30,20,0,0] #init pop

#Summary Stats 

beta = 0.1

net_birth = 0.003599999938160181

carrying = 5000

exposed_rate = 0.1666666716337204

recovery_rate = 0.125

omega = 0.5

death = 0.95

decay = 60

wane = 0.0055555556900799274 #180 days

sigma = 0.75

birth_width = 3

birth_offset = 75

birth_amp = 0.009801327250897884

seasonal_day = 30

seasonal_offset = 0 

model_number = 4

  
    
params = [beta, net_birth, carrying, exposed_rate, recovery_rate, omega, death, decay, wane, sigma, birth_width, birth_offset, birth_amp, seasonal_day, seasonal_offset, model_number]

#Our observations, 3 summary params their means and std (found from calculating 95\% conf interval)

mean_ep = 1.5 
std_ep = 0.604

mean_pd = 75
std_pd = 6.08

mean_mt = 180
std_mt = 36.475

observation = [mean_ep, mean_pd, mean_mt]


#Transition Matrix

const nt = 1 #total number of groups and/or farms
const nc = 5 #number of classes (SEIRC)
const eqs = 11 #number of processes

#Matrix of all the transitions between classes for Gillespie model
const dc = sparse(zeros(nc,eqs))

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
function dummy()
    input_path = "/home/callum/ASF/Inputs/"; #path to model data
    ASF_Inputs.Model_Data(input_path)
    println("it lives")
    end

function distance(Y,Y0)
    #Distance function for ABC fitting
    U0 = Y0["SS"] 
    U = Y["SS"]

    vals = (U0 .- U) ./ [std_ep, std_pd, std_mt]
    
    
    d = sqrt(sum(vals.^2))
    
    return d
end

function asf_model_one_group(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, bw, bo, k, la, lo, ty  = p 
    
    u[u.<0] .= 0
    
    S, E, I, R, C = u
    N = sum(u)
    L = S + E + I + R
    if N == 0 
        N = 1
    end
    if ty == 1
        beta_mod = 1
    elseif ty == 2
        beta_mod = L/K
    elseif ty == 3
        beta_mod =  tanh(1.2 *L/K - 1.2 ) + 1
    else
        beta_mod = sqrt(L/K)
    end
    
    Deaths = μ_p*(σ + ((1-σ))*sqrt(L/K))
   
    Lambda = λ + la * cos((t + lo) * 2*pi/365)

    p_mag = birth_pulse_vector(t,k,bw,bo)
    Births = p_mag*(σ *L + ((1-σ))*sqrt(L*K))
    
   #11 processes
    out[1] = Births
    out[2] = S * Deaths
    out[3] = beta_mod * β * (I + ω * C) * S / N
    out[4] = E * Deaths
    out[5] = ζ * E
    out[6] = ρ * γ * I 
    out[7] = I * Deaths
    out[8] = γ * (1 - ρ) * I
    out[9] = R * Deaths
    out[10] = (1 / (Lambda)) * C
    out[11] = κ * R 


    nothing
end

function run_analysis(sol)
       #function to analyse model output (converts classes into more bitesize form)
        data = reduce(vcat,transpose.(sol.u))
        data[data .< 0 ] .= 0
   
        s_d = data[:,1]
        e_d = data[:,2]
        i_d = data[:,3]
        r_d = data[:,4]
        c_d = data[:,5]
 
        disease_total = e_d + i_d + c_d #classes with disease,
        disease_alive = e_d + i_d
 
        disease_free = s_d + r_d #classes without disease,
 
        disease_sum = sum(disease_total,dims=2)
        disease_alive_sum =  sum(disease_alive,dims=2)
        disease_free_sum = sum(disease_free,dims=2)
        population_sum = disease_alive_sum + disease_free_sum;
   
        return disease_sum, disease_alive_sum, disease_free_sum, population_sum
end
    
function summary_stat(solution)
    #takes the model output and converts it into the needed three summary statistics
   
    filter = true
    ep = 0
    mt = 0
    pd = 0
    
    detection_p = 0.05
    pop_K = 5100
    
    starting_p = detection_p*pop_K
   
    d, da,f,p = run_analysis(solution)
    ln = length(da[3*365:end])
    ep = 100*sum(da[3*365:end])/sum(p[3*365:end])
    pd = 100*(1-(sum(p[3*365:end])/ln)/pop_K)

    max_d = findmax(d)[2][1]
    
    
    if filter
        output = zeros(4)
        if d[end] == 0
            output[1] = 0
            output[2] = 0
            output[3] = 0
            output[4] = 0
        else
            if maximum(d) <= starting_p
                take_off_time = 0
            else
                take_off_time = findfirst(>(starting_p), d)[1]
            end

            mt = max_d-take_off_time

            output[1] = ep
            output[2] = pd
            output[3] = mt 
            output[4] = 1
        end
    else
        output = zeros(3)        

        if maximum(d) <= starting_p
            take_off_time = 0
        else
            take_off_time = findfirst(>(starting_p), d)[1]
        end

        mt = max_d-take_off_time

        output[1] = ep
        output[2] = pd
        output[3] = mt
    
    end
  
    
    return output
        
end

function birth_pulse_vector(t,k,s,p)
    #birth pulse not for a vector
    return k*exp(-s*cos(pi*(t+p)/365)^2)
end

function model_1(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 1
    
    rj = RegularJump(asf_model_one_group, regular_c, eqs)
    prob = DiscreteProblem(u0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)

    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_2(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 2
    
    rj = RegularJump(asf_model_one_group, regular_c, eqs)
    prob = DiscreteProblem(u0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_3(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 3
    
    rj = RegularJump(asf_model_one_group, regular_c, eqs)
    prob = DiscreteProblem(u0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_4(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 4
    
    rj = RegularJump(asf_model_one_group, regular_c, eqs)
    prob = DiscreteProblem(u0,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end



end
