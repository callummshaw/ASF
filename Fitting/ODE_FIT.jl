module SIR_ODE

using DifferentialEquations

function asf_model_ode(du,u,p,t)
    #ode equivelent of our ASF model
    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, bw, bo, k, la, lo, ty  = p 
    
    S, E, I, R, C = u
    N = sum(u)
    L = S + E + I + R
    
    #the different contact functions
    if ty == 1
        beta_mod = 1
    elseif ty == 2
        beta_mod = L/K
    elseif ty == 3
        beta_mod =  tanh(1.2 *L/K - 1.2 ) + 1
    else
        beta_mod = sqrt(L/K)
    end
    
    ds = μ_p*(σ + ((1-σ))*sqrt(L/K))
    
    du[1] = k*exp(-bw*cos(pi*(t+bo)/365)^2)*(σ .* L .+ ((1-σ)) .* sqrt.(L .* K)) + κ*R - ds*S - beta_mod*β*(I + ω*C)*S/N
    du[2] = beta_mod*β*(I + ω*C)*S/N - (ds + ζ)*E
    du[3] = ζ*E - (ds + γ)*I
    du[4] = γ*(1-ρ)*I - (ds + κ)*R
    du[5] = (ds + γ*ρ)*I -1/( λ + la * cos((t + lo) * 2*pi/365))*C
    
    nothing
end

#Tspan params
start_day = 180.0
n_years = 6

#seeding init pop
n_exp = 30 #number of exposed
n_inf = 20 #number of infected
N_total = 6518 # total population at start date (180) found from previous burn in
u0 = [N_total-n_exp-n_inf,30,20,0,0] #init pop

#Summary Stats 

params = [0.1, 0.003599999938160181, 5000.0, 0.1666666716337204,0.125, 0.5, 0.949999988079071, 60.0, 0.0055555556900799274, 0.75, 3.0, 75.0, 0.009801327250897884, 30.0, 0.0, 4]


mean_ep = 1.5
std_ep = 0.604

mean_pd = 75
std_pd = 6.08

mean_mt = 180
std_mt = 36.475

observation = [mean_ep, mean_pd, mean_mt]


function distance(Y,Y0)
    
    U0 = Y0["SS"] 
    U = Y["SS"]

    vals = (U0 .- U) ./ [std_ep, std_pd, std_mt]
    
    
    d = sqrt(sum(vals.^2))
    
    return d
end
    
function run_analysis(sol)
   
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
    
    output = zeros(3)
   
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

    if maximum(d) <= starting_p
        take_off_time = 0
    else
        take_off_time = findfirst(>(starting_p), d)[1]
    end

    mt = max_d-take_off_time
        
        
    output[1] = ep
    output[2] = pd
    output[3] = mt
        
    return output
        
end

function model_1(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 1
    
    prob_ode = ODEProblem(asf_model_ode, u0, (start_day,n_years*365.0+start_day), params)
    sol = solve(prob_ode, saveat = 1,reltol=1e-8)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_2(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 2
    
    prob_ode = ODEProblem(asf_model_ode, u0, (start_day,n_years*365.0+start_day), params)
    sol = solve(prob_ode, saveat = 1,reltol=1e-8)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_3(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 3
    
    prob_ode = ODEProblem(asf_model_ode, u0, (start_day,n_years*365.0+start_day), params)
    sol = solve(prob_ode, saveat = 1,reltol=1e-8)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

function model_4(par)
    
    p1 = par["p1"]
    p2 = par["p2"]
    
    params[1] = p1
    params[6] = p2
    
    params[end] = 4
    
    prob_ode = ODEProblem(asf_model_ode, u0, (start_day,n_years*365.0+start_day), params)
    sol = solve(prob_ode, saveat = 1,reltol=1e-8)
        
    summary = summary_stat(sol)
    
    return Dict("SS"=>summary)
    
end

end
