module SIR_ODE

using DifferentialEquations

function asf_model_ode(du,u,p,t)
    #ode equivelent of our ASF model
    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, bw, bo, k, la, lo, ty, ttt  = p 
    
    S, E, I, R, C = u
    N = sum(u)
    L = S + E + I + R
    
    #the different contact functions
    if ty == 1
        beta_mod = 1
    elseif ty == 2
        beta_mod = L/K
    elseif ty == 3
        beta_mod =  tanh(1.5 *L/K - 1.5 ) + 1
    else            
        beta_mod = sqrt(L/K)
    end
    
    ds = μ_p*(σ + ((1-σ))*sqrt(L/K))
    
    if t < int_start #prior to intervention
        lam = 1/( λ + la * cos((t + lo) * 2*pi/365))
    else #intervention scaling of decay time
        lam = ttt*1/(( λ + la * cos((t + lo) * 2*pi/365)))
    end
    
    du[1] = k*exp(-bw*cos(pi*(t+bo)/365)^2)*(σ .* L .+ ((1-σ)) .* sqrt.(L .* K)) + κ*R - ds*S - beta_mod*β*(I + ω*C)*S/N
    du[2] = beta_mod*β*(I + ω*C)*S/N - (ds + ζ)*E
    du[3] = ζ*E - (ds + γ)*I
    du[4] = γ*(1-ρ)*I - (ds + κ)*R
    du[5] = (ds + γ*ρ)*I -lam*C
    
    nothing
end
function asf_model_logistic(du,u,p,t)
    #ode equivelent of our ASF model but LOGISTIC!
    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, bw, bo, k, la, lo, ty,ttt  = p 
    
    S, E, I, R, C = u
    N = sum(u)
    L = S + E + I + R
    
    #the different contact functions
    if ty == 1
        beta_mod = 1
    elseif ty == 2
        beta_mod = L/K
    elseif ty == 3
        beta_mod =  tanh(1.5 *L/K - 1.5 ) + 1
    elseif ty == 4
        beta_mod = sqrt(L/K)
    end
    
    br =  k*exp(-bw*cos(pi*(t+bo)/365)^2)
    μ_p = (2/3)*μ_p
    r = br - μ_p
    g = r/K
    
    if t < int_start #prior to intervention
        lam = 1/( λ + la * cos((t + lo) * 2*pi/365))
    else #intervention scaling of decay time
        lam =ttt* 1/( λ + la * cos((t + lo) * 2*pi/365))
    end
    
    
    du[1] = br*L -(μ_p+g*L)*S + κ*R - beta_mod*β*(I + ω*C)*S/N
    du[2] = beta_mod*β*(I + ω*C)*S/N - (μ_p+g*L+ζ)*E
    du[3] = ζ*E - (μ_p + g*L + γ)*I
    du[4] = γ*(1-ρ)*I - (μ_p+g*L + κ)*R
    du[5] = (μ_p+g*L + γ*ρ)*I -lam*C
    
    nothing
end

#Time till intervention starts!
int_start = 0*30


#Tspan params
start_day = 180.0
n_years = 3

#seeding init pop
n_exp = 30 #number of exposed
n_inf = 20 #number of infected
N_total = 6518 #5000 for logistic # total population at start date (180) found from previous burn in
u0 = [N_total-n_exp-n_inf,n_exp,n_inf,0,0] #init pop

#Summary Stats 

#beta base
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

intervention_strength = 1.0 
   
    
params = [beta, net_birth, carrying, exposed_rate, recovery_rate, omega, death, decay, wane, sigma, birth_width, birth_offset, birth_amp, seasonal_day, seasonal_offset, model_number, intervention_strength]


mean_ep = 1.5
std_ep = 0.604

mean_pd = 75
std_pd = 6.08

mean_mt = 180
std_mt = 36.475

observation = Dict("SS"=>[mean_ep, mean_pd, mean_mt])

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
    
    filter = true #turn on for intervention sim!
    ep = 0
    mt = 0
    pd = 0
    
    detection_p = 0.05
    pop_K = 5000
    
    starting_p = detection_p*pop_K
   
    d, da,f,p = run_analysis(solution)
    ln = length(da[3*365:end])
    ep = 100*sum(da[3*365:end])/sum(p[3*365:end])
    pd = 100*(1-(sum(p[3*365:end])/ln)/pop_K)

    max_d = findmax(d)[2][1]
    
    if filter
        output = zeros(4)
        if d[end] < 0.25
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


function model_int(par)
    
    force = false #not looking at force of infection
    
    p1 = par[1]#beta
    p2 = par[2]#omega
    p3 = par[3]#model number
    p4 = par[4]#intervention
    
    
    base = copy(decay)
    params[1] = p1
    params[6] = p2
    params[end-1] = p4
    params[end] = p3
    
    
    
    
    prob_ode = ODEProblem(asf_model_ode, u0, (start_day,n_years*365.0+start_day+int_start), params)
    sol = solve(prob_ode, saveat = 1,reltol=1e-8)
        
    
    if force
        data = reduce(vcat,transpose.(sol.u))
        data[data .< 0 ] .= 0

        s_d = data[:,1]
        e_d = data[:,2]
        i_d = data[:,3]
        r_d = data[:,4]
        c_d = data[:,5]

        n_d = s_d+e_d+i_d+r_d

        return s_d, n_d
        
    else
        summary = summary_stat(sol)
    
        return summary[4] #if endemic! this is for intervention sims
        
    end

    
end



end
