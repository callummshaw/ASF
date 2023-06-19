"Module that contains the three ASF models, ODE, homogeneous tau-leaping, and heterogeneous tau-leaping"

module Models 

using LinearAlgebra
using Distributions
using SparseArrays
using Random

export ASF_M3_full
export ASF_M3_single
export ASF_M2
export ASF_M1

BLAS.set_num_threads(8)
using Octavian 

function ASF_M3_full(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!
 
    ref_density = 2.8 #baseline density (from Baltics where modelled was fitted)
    year = 365 #days in a year

    u[u.<0] .= 0
   
    pop_data = p.Populations
    deff_s = zeros(Float32,pop_data.pop)
    for i in 1:pop_data.pop #looping through populations!
        
        s0 = pop_data.cum_sum[i] #start index of pop
        ei = pop_data.cum_sum[i+1] #end index of pop

        S = Vector{UInt8}(u[5*s0+1:5:5*ei])
        E = Vector{UInt8}(u[5*s0+2:5:5*ei])
        I = Vector{UInt8}(u[5*s0+3:5:5*ei])
        R = Vector{UInt8}(u[5*s0+4:5:5*ei])
        C = Vector{UInt8}(u[5*s0+5:5:5*ei])
        
        Np = S + E + I + R
        Nt = Np + C

        Nt[Nt .== 0] .= 1

        K = p.K[i]

        tg = length(Np) #total groups in all populations
        tp = sum(Np) # total living pigs
        
        d_eff = sqrt((tp / pop_data.area[i]) / ref_density) #account for different densities between fitting and current pop
        deff_s[i] = d_eff
       
        p_mag = @. p.k[i]*exp(-p.bw[i]*cos.(pi*(t+p.bo[i])/365)^2) #birth pulse value at time t
        Deaths = @. p.μ_p[i]*(0.75 + ((1-0.75))*sqrt(Np/K))*p.g[i] #rate
        Births = @. p_mag*(0.75 * Np + ((1-0.75)) * sqrt(Np * K))#total! (rate times NP)

        #now stopping boar births
        mask_boar = (K .== 1) .& (Np .> 0) #boars with a positive population
        boar_births = p_mag*sum(mask_boar)
        Births[mask_boar] .= 0
        mask_p_s = (Np .> 1) .& (K .> 1) #moving it to postive sow groups with at least 2 pigs
        Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
        
        n_empty  = sum(Np .== 0 )   
        
        if n_empty/tg > 0.01   #migration births (filling dead nodes if there is a connecting group with 2 or more pigs)
            
            n_r = (n_empty/tg)^2 #squared to reduce intesity
            

            t2 .= Np
            t2[t2 .< 2] .= 0
            t1 .= nets * t2

            dd = copy(Np)
            dd[dd .< 2] .= 0
            connected_pops = pop_data.networks[i] * dd

            #Groups with 3 or more pigs can have emigration
            mask_em =  (dd .> 0) #populations that will have emigration

            em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

            mask_im = (Np .== 0) .& (connected_pops .> 1) #population zero but connected groups have 5 or more pigs

            Births[mask_em] .*= n_r
            Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)

        end 
      
        if sum(E+I+C) == 0#no disease in populations

            out[11*s0+1:11:11*ei] .= Births
            out[11*s0+2:11:11*ei] .= @. S*Deaths
            out[11*s0+3:11:11*ei] .= 0
            out[11*s0+4:11:11*ei] .= 0
            out[11*s0+5:11:11*ei] .= 0
            out[11*s0+6:11:11*ei] .= 0 
            out[11*s0+7:11:11*ei] .= 0
            out[11*s0+8:11:11*ei] .= 0
            out[11*s0+9:11:11*ei] .= @. R*Deaths
            out[11*s0+10:11:11*ei].= 0
            out[11*s0+11:11:11*ei].= @. p.κ[i]*R 
        else
            
            Lambda = @. p.λ[i] + p.la[i] * cos((t + p.lo[i]) * 2*pi/year) #decay
            out[11*s0+1:11:11*ei] .= Births
            out[11*s0+2:11:11*ei] .= @. S*Deaths
            out[11*s0+3:11:11*ei] .= p.β_i[i] .* (S ./ Nt) .* (I .+ p.ω[i] .* C) #intra
            out[11*s0+4:11:11*ei] .= @. E*Deaths
            out[11*s0+5:11:11*ei] .= @. p.ζ[i] *E
            out[11*s0+6:11:11*ei] .= @. p.ρ[i]*p.γ[i]*I 
            out[11*s0+7:11:11*ei] .= @. I*Deaths
            out[11*s0+8:11:11*ei] .= @. p.γ[i]*(1-p.ρ[i])*I
            out[11*s0+9:11:11*ei] .= @. R*Deaths
            out[11*s0+10:11:11*ei].= @. (1/Lambda)*C
            out[11*s0+11:11:11*ei].= @. p.κ[i]*R 
            
            for con in eachrow(pop_data.inter_connections[i]) #intra
                g1 = con[1]
                g2 = con[2]

                NTT = Nt[g1] + Nt[g2]

                #Pop 2 to 1
                out[11*(s0+g1-1)+3] += d_eff * S[g1] * p.β_o[i][g1] * ( I[g2] + p.ω[i][g2] * C[g2]) / NTT

                #Pop 1 to 2
                out[11*(s0+g2-1)+3] += d_eff * S[g2] * p.β_o[i][g2] * ( I[g1] + p.ω[i][g1] * C[g1]) / NTT

            end
        end
    end 

    #now we need to handle the between population transmission!
    for con in eachrow(pop_data.pop_connections) #looping through all inter population connections
        g1 = con[1]
        g2 = con[2]
        
        p1 = con[3]
        p2 = con[4]

        dm = (deff_s[p1] + deff_s[p2])/2 
        β = (p.β_o[p1][g1]+p.β_o[p2][g2])/2

        s01 = pop_data.cum_sum[p1]
        s02 = pop_data.cum_sum[p2]
        
        if (u[5*(g1+s01-1)+3] + u[5*(g1+s01-1)+5] + u[5*(g2+s02-1)+3] + u[5*(g2+s02-1)+5]) > 0 #INFECTIONS Can Occur!
            NTT = sum(u[5*(g1+s01-1)+1:5*(g1+s01-1)+5]) + sum(u[5*(g2+s02-1)+1:5*(g2+s02-1)+5])
            
            #Pop 2 to 1
            out[11*(g1+s01-1)+3] += dm * u[5*(g1+s01-1)+1] * β * ( u[5*(g2+s02-1)+3] + p.ω[p2][g2] * u[5*(g2+s02-1)+5]) / NTT
            
            #Pop 1 to 2
            out[11*(g2+s02-1)+3] += dm * u[5*(g2+s02-1)+1] * β * ( u[5*(g1+s01-1)+3] + p.ω[p1][g1] * u[5*(g1+s01-1)+5]) / NTT
            
        end
    end

    nothing
end

function ASF_M3_single(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β_o, β_i, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, g, bw, bo, k, la, lo, area, cons, nets, t1, t2 = p
    
    ref_density = 2.8 #baseline density (from Baltics where modelled was fitted)
    year = 365 #days in a year

    u[u.<0] .= 0
   
    S = Vector{UInt8}(u[1:5:end])
    E = Vector{UInt8}(u[2:5:end])
    I = Vector{UInt8}(u[3:5:end])
    R = Vector{UInt8}(u[4:5:end])
    C = Vector{UInt8}(u[5:5:end])
  
    Np = S .+ E .+ I .+ R
    Nt = Np .+ C
    
    Nt[Nt .== 0] .= 1
    
    tg = length(Np) #total groups in all populations
    tp = sum(Np) # total living pigs
    
    d_eff = sqrt((tp / area) / ref_density) #account for different densities between fitting and current pop
    
    Lambda = @. λ + la * cos((t + lo) * 2*pi/year) #decay
    p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
   
    Deaths = @. μ_p*(σ + ((1-σ))*sqrt(Np./K))*g #rate
    Births = @. p_mag*(σ * Np + ((1-σ)) * sqrt(Np .* K))#total! (rate times NP)

    #now stopping boar births
    mask_boar = (K .== 1) .& (Np .> 0) #boars with a positive population
    boar_births = p_mag*sum(mask_boar)
    Births[mask_boar] .= 0
    mask_p_s = (Np .> 1) .& (K .> 1) #moving it to postive sow groups with at least 2 pigs
    Births[mask_p_s] .+= boar_births ./ sum(mask_p_s) 
    
    n_empty  = sum(Np .== 0 )   
     
     if n_empty/tg > 0.01   #migration births (filling dead nodes if there is a connecting group with 2 or more pigs)
            
            n_r = (n_empty/tg)^2 #squared to reduce intesity
            
            t2 .= Np
            t2[t2 .< 2] .= 0
            t1 .= nets * t2
            
            #Groups with 3 or more pigs can have emigration
            mask_em =  (t2 .> 0) #populations that will have emigration

            em_force = sum(Births[mask_em]) #"extra" births in these populations that we will transfer

            mask_im = (Np .== 0) .& (t1 .> 1) #population zero but connected groups have 5 or more pigs

            Births[mask_em] .*= n_r
            Births[mask_im] .= (1 - n_r)*em_force/sum(mask_im)

     end 
    
    
    out[3:11:end] .= β_i .* (S ./ Nt) .* (I .+ ω .* C) #intra
    
    for con in eachrow(cons) #intra
        g1 = con[1]
        g2 = con[2]

        if (I[g1]+I[g2]+C[g1]+C[g2]) > 0 #INFECTIONS Can Occur!
            
            NTT = Nt[g1] + Nt[g2]
    
            #Pop 2 to 1
            out[11*(g1-1)+3] += d_eff * S[g1] * β_o[g1] * ( I[g2] + ω[g2] * C[g2]) / NTT
            
            #Pop 1 to 2
            out[11*(g2-1)+3] += d_eff * S[g2] * β_o[g2] * ( I[g1] + ω[g1] * C[g1]) / NTT

        end
       
    end
    
     out[1:11:end] .= Births
     out[2:11:end] .= @. S*Deaths
     out[4:11:end] .= @. E*Deaths
     out[5:11:end] .= @. ζ*E
     out[6:11:end] .= @. ρ*γ*I 
     out[7:11:end] .= @. I*Deaths
     out[8:11:end] .= @. γ*(1-ρ)*I
     out[9:11:end] .= @. R*Deaths
     out[10:11:end].= @. (1/Lambda)*C
     out[11:11:end].= @. κ*R 
   
    nothing
end

function ASF_M2(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, Seasonal, bw, bo, k, la, lo, Area = p 

    ref_density = 2.8
    
    u[u.<0] .= 0
    
    S, E, I, R, C = u
    Np = S + E + I + R
    Nt = Np + C 
    
    if Nt == 0 
        Nt = 1
    end
   
    beta_mod = sqrt((Np/Area)/ref_density)

    Lambda = λ + la * cos.((t + lo) * 2*pi/365) #decay
    p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
 
    Deaths = μ_p*(σ + ((1-σ))* sqrt(Np/K)) #rate
    Births = p_mag*(σ * Np + ((1-σ)) * sqrt(Np)*sqrt(K))#total! (rate times NP)
    
    #11 processes
    out[1] = Births
    out[2] = S * Deaths
    out[3] = beta_mod * β * (I + ω * C) * S / Nt
    out[4] = E * Deaths
    out[5] = ζ * E
    out[6] = ρ * γ * I 
    out[7] = I * Deaths
    out[8] = γ * (1 - ρ) * I
    out[9] = R * Deaths
    out[10] = (1 / Lambda) * C
    out[11] = κ * R 
   
    nothing
end


function ASF_M1(du,u,p,t)
    
    #ode equivelent of our ASF model
    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, Seasonal, bw, bo, k, la, lo, Area = p 
    
    ref_density = 2.8
    
    S, E, I, R, C = u
    Np = S + E + I + R
    Nt = N + C 
    
    if Nt == 0 
        Nt = 1
    end
   
    beta_mod = sqrt((Np/Area)/ref_density)
    
    
    #if Seasonal    
    Lambda = λ + la * cos((t + lo) * 2*pi/365) #decay
    p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
    #else 
     #   Lambda = λ
      #  p_mag = μ_p
    #end

    #if θ == 0.5 #density births and death!
    ds = μ_p*(σ+ ((1-σ))*beta_mod) #rate
    Births = p_mag*(σ * Np + ((1-σ)) * sqrt(Np * K))#total! (rate times NP)
   # else
     #   ds = μ_p.*(σ .+ ((1-σ)).*(Np./K).^θ) #rate
      #  Births = p_mag.*(σ .* Np .+ ((1-σ)) .* Np.^(1-θ) .* K.^θ)#total! (rate times NP)
    #end
    
    du[1] = Births + κ*R - ds*S - beta_mod*β*(I + ω*C)*S/Nt
    du[2] = beta_mod*β*(I + ω*C)*S/Nt - (ds + ζ)*E
    du[3] = ζ*E - (ds + γ)*I
    du[4] = γ*(1-ρ)*I - (ds + κ)*R
    du[5] = (ds + γ*ρ)*I -(1/Lambda)*C
    
    nothing
end

function birth_pulse(t, p, i)
    #birth pulse for a vector!
    return p.k*exp(-p.bw[i]*cos(pi*(t+p.bo[i])/365)^2)
end

function birth_pulse_vector(t,k,s,p)
    #birth pulse not for a vector
    return k*exp(-s*cos(pi*(t+p)/365)^2)
end

end
