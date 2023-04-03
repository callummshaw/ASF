module SIR_TAU_I

using DifferentialEquations
using LinearAlgebra
using SparseArrays
using Distributions
using Random

include("/home/callum/ASF/Modules/ASF_input.jl");

#Tspan params
start_day = 180
n_years = 3
int_start = 3*30

Tspan = (start_day,n_years*365+start_day+int_start)
n_con = 6
#Summary Stats 
#Our observations, 3 summary params their means and std (found from calculating 95\% conf interval)

mean_ep = 1.5
std_ep = 0.604

mean_pd = 75
std_pd = 6.08

mean_mt = 180
std_mt = 36.475

observation = Dict("SS"=>[mean_ep, mean_pd, mean_mt])

KT = 5000

const nt = 1000
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

    vals = (U0 .- U) ./ [std_ep, std_pd, std_mt]
    
    
    d = sqrt(sum(vals.^2))
    
    return d
end

function asf_model_one(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β_i, β_o, β_b, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, Seasonal, bw, bo, k, la, lo, Area, ty, int_str = p 
    ref_density = 1 #baseline density (from Baltics where modelled was fitted)
    year = 365 #days in a year

    u[u.<0] .= 0
   
    S = Vector{UInt8}(u[1:5:end])
    E = Vector{UInt8}(u[2:5:end])
    I = Vector{UInt8}(u[3:5:end])
    R = Vector{UInt8}(u[4:5:end])
    C = Vector{UInt8}(u[5:5:end])

    N = S .+ E .+ I .+ R .+ C
    Np = S .+ E .+ I .+ R
    
    N[N .== 0] .= 1
    
    tg = length(Np) #total groups in all populations
    tp = sum(Np) # total living pigs

    if ty == 1
        beta_mod = 1
    elseif ty == 2
        beta_mod = tp/KT
    elseif ty == 3
        beta_mod =  tanh.(1.2 *tp/KT - 1.2 ) + 1
    else
        beta_mod = sqrt.(abs.(tp/KT))
    end
    
    Deaths = μ_p.*(σ .+ ((1-σ)).*sqrt.(abs.(Np))./sqrt.(abs.(K)))*g
    
    if t < int_start
        Lambda = λ + la * cos.((t + lo) * 2*pi/year)
    else
        Lambda = int_str*(λ + la * cos.((t + lo) * 2*pi/year))
    end
    
    p_mag = birth_pulse_vector(t,k,bw,bo)
    Births = p_mag.*(σ .* Np .+ ((1-σ)) .* sqrt.(abs.(Np .* K)))#Np.^(1-θ) .* K.^θ)
    
    #now stopping boar births
    mask_boar = (K .== 1) .& (Np .> 0) #boars with a positive population
    boar_births = p_mag*sum(mask_boar)
    Births[mask_boar] .= 0
    mask_p_s = (Np .> 1) .& (K .> 1) #moving it to postive 
    Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
     
    
    n_empty  = sum(Np .== 0 )   
    
    if n_empty/tg > 0.01   #migration births (filling dead nodes if there is a connecting group with 2 or more pigs)
        
        n_r = (n_empty/tg)^2
        
        dd = copy(Np)
        dd[dd .< 2] .= 0
        connected_pops = β_b * dd

            #Groups with 3 or more pigs can have emigration
        mask_em =  (dd .> 0) #populations that will have emigration

        em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

        mask_im = (Np .== 0) .& (connected_pops .> 1) #population zero but connected groups have 5 or more pigs

        Births[mask_em] .*= n_r
        Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)

    end
    
    v = ones(Int8,tg)
        
    populations  = v*N'+ N*v'
    
    populations[diagind(populations)] = N

    out[1:11:end] .= Births
    out[2:11:end] .= S.*Deaths
    out[3:11:end] .=  (((beta_mod .* β_o .* S) ./ populations)*(I .+ ω .* C)).+ β_i .* (S ./ N) .* (I .+ ω .* C)
    out[4:11:end] .= E.*Deaths
    out[5:11:end] .= ζ .* E
    out[6:11:end] .= ρ .* γ .* I 
    out[7:11:end] .= I.*Deaths
    out[8:11:end] .= γ .* (1 .- ρ) .* I
    out[9:11:end] .= R.*Deaths
    out[10:11:end].= (1 ./ Lambda) .* C
    out[11:11:end] .= κ .* R 


    nothing
end

function run_analysis(sol)
       #function to analyse model output (converts classes into more bitesize form)
        data = reduce(vcat,transpose.(sol.u))
        data[data .< 0 ] .= 0
   
        s_d = data[:,1:5:end]
        e_d = data[:,2:5:end]
        i_d = data[:,3:5:end]
        r_d = data[:,4:5:end]
        c_d = data[:,5:5:end]
 
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

function burn_in_pop(params, U0)
    
    Burn_U0 =  copy(U0)
    Burn_U0[2:5:end] .= 0
    Burn_U0[3:5:end] .= 0
    
    rj_burn = RegularJump(asf_model_pop, regular_c, eqs*nt)
    prob_burn = DiscreteProblem(Burn_U0,(0.0,275),params)
    jump_prob_burn = JumpProblem(prob_burn, Direct(), rj_burn)
    sol_burn = solve(jump_prob_burn, SimpleTauLeaping(),dt=1);

    U_burn = copy(sol_burn[params[18]+start_day]); #population at start date
        
    rr = rand(1:nt) #seeding diease in starting pop
    ra = rr -1
    if U_burn[ra*5+1] > 1
        U_burn[ra*5+1] = 0
        U_burn[ra*5+2] = 3
        U_burn[ra*5+3] = 2
    else
        U_burn[ra*5+1] = 0
        U_burn[ra*5+3] = 1
    end

    netw = params[3][:,rr] #related populations
    cons = findall(>(0), netw)
    
    if length(cons) <= 4
        wanted = cons
    else
        wanted=shuffle(cons)[1:4] #seeding in 4 other pops so 5 in total!
    end 
    
    for i in cons
        i1 = i -1
        if U_burn[i1*5+1] > 1
            U_burn[i1*5+1] = 0
            U_burn[i1*5+2] = 3
            U_burn[i1*5+3] = 2
        else
            U_burn[i1*5+1] = 0
            U_burn[i1*5+3] = 1
        end

    end
    
    return U_burn
end

function model_int(par, type_net)
    
    
    p1 = par[1]
    p2 = par[2]
    p3 = par[3]
    
    trans_type = par[5]
    int_strength = par[4]
    
    if type_net == "r"
        input_path = "/home/callum/ASF/Inputs_r/"; #path to model data
    elseif type_net == "sf"
        input_path = "/home/callum/ASF/Inputs_sw/"; #path to model data
    elseif  type_net == "sw"
        input_path = "/home/callum/ASF/Inputs_sf/"; #path to model data
    end
    
    input = ASF_Inputs.Model_Data(input_path)
    
    params = convert(input.Parameters)
    
    U0 = input.U0

    init_pop = burn_in_pop(params, U0)
    

    #beta
    params[1] .= p1 #intra
    params[2][params[2] .!= 0 ] .= p2/n_con #inter
    
    #corpse
    params[8] = p3 
    
    params[23] = trans_type
    params[24] = int_strength
    
    rj = RegularJump(asf_model_one, regular_c, eqs*nt)
    prob = DiscreteProblem(init_pop,Tspan,params)
    jump_prob = JumpProblem(prob,Direct(),rj)
    sol = solve(jump_prob, SimpleTauLeaping(), dt =1)
        
    summary = summary_stat(sol)
    
    return summary[4]
    
end



function convert(input)
    #Function to convert input structure to simple array (used mainly for fitting)
    params = Vector{Any}(undef,24)
    
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
    
    params[23] = 1 #model number
    params[24] = 1 #intervention strength
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

end
