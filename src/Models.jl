"Module that contains the three ASF models, ODE, homogeneous tau-leaping, and heterogeneous tau-leaping"

module Models 

using LinearAlgebra
using Distributions
using SparseArrays

export ASF_M3
export ASF_M3S
export ASF_M3SL
export ASF_M2
export ASF_M1

BLAS.set_num_threads(8)
using Octavian 

function ASF_M3(out,u,p,t)

    ref_density = 1 #baseline density (from Baltics where modelled was fitted)
    offset = 180 #seeding in the summer!
    year = 365 #days in a year

    u[u.<0] .= 0
    
    S = Vector{UInt8}(u[1:5:end])
    E = Vector{UInt8}(u[2:5:end])
    I = Vector{UInt8}(u[3:5:end])
    R = Vector{UInt8}(u[4:5:end])
    C = Vector{UInt8}(u[5:5:end])
    
    
    S = @view u[1:5:end]
    E = @view u[2:5:end]
    I = @view u[3:5:end]
    R = @view u[4:5:end]
    C = @view u[5:5:end]

    N = S .+ E .+ I .+ R .+ C
    Np = S .+ E .+ I .+ R
    
    N[N .== 0] .= 1
    
    Pops = p.Populations 
    
    tp = Pops.cum_sum[end] #total groups in all populations

    beta = copy(p.β)
    Births = copy(p.μ_p)
    Deaths = copy(p.μ_p)
    Lambda = copy(p.λ)
    connected_pops = p.β_b * Np

    for i in 1:Pops.pop #going through populations
    
        nf = Pops.feral[i] #number of feral groups in region
        ncs = Pops.cum_sum[i] #cumsum of farm and ferals over all regions
        N_feral = sum(Np[ncs+1:ncs+nf]) #total feral population in region i
        Density = N_feral/Pops.area[i]

        beta[p.β_d .== i] .*= ((Density/ref_density).^(p.η))
        Deaths[ncs+1:ncs+nf] *= (p.σ[i] .+ (1-p.σ[i]).*Np[ncs+1:ncs+nf].^p.θ[i].*p.K[ncs+1:ncs+nf].^(-p.θ[i]))
        
        if p.Seasonal

            p_mag = birth_pulse(t, p,i)
            Births[ncs+1:ncs+nf] = p_mag.*(p.σ[i] .* Np[ncs+1:ncs+nf] .+ (1-p.σ[i]).*Np[ncs+1:ncs+nf].^(1-p.θ[i]).*p.K[ncs+1:ncs+nf].^p.θ[i])
            Lambda[ncs+1:ncs+nf] .+= p.la[i] * cos((t + offset + p.lo[i]) * 2*pi/year)
        
            #now stopping boar births
            mask_boar = (p.K[ncs+1:ncs+nf] .== 1) .& (Np[ncs+1:ncs+nf] .> 0)
            boar_births = sum(mask_boar)
            Births[mask_boar] .= 0
            mask_p_s = (Np[ncs+1:ncs+nf] .> 1) .& (p.K[ncs+1:ncs+nf] .> 1)
            Births[mask_p_s] .+= p_mag*boar_births ./ sum(mask_p_s) 

            if  p_mag > mean(p.μ_p[ncs+1:ncs+nf])
                mask_em =  (Np[ncs+1:ncs+nf] .> 3) .& (p.K[ncs+1:ncs+nf] .> 1)
                mask_im = (Np[ncs+1:ncs+nf] .== 0) .& (connected_pops[ncs+1:ncs+nf] .> 3) #population zero but connected groups have 3 or more pigs
                extra_b = sum(Births[mask_im] .= 3*p_mag)
                Births[mask_em] .-= extra_b ./ sum(mask_p_s)
            end
        else
            Births[ncs+1:ncs+nf] *= (p.σ[i] .* Np[ncs+1:ncs+nf] .+ (1-p.σ[i]).*Np.^(1-p.θ[i]).*K[ncs+1:ncs+nf].^p.θ[i])

            mask_boar = (p.K[ncs+1:ncs+nf] .== 1) .& (Np[ncs+1:ncs+nf] .> 0)
            boar_births = sum(mask_boar)
            Births[mask_boar] .= 0
            mask_p_s = (Np[ncs+1:ncs+nf] .> 1) .& (p.K[ncs+1:ncs+nf] .> 1)
            Births[mask_p_s] .+= μ_p*boar_births ./ sum(mask_p_s) 

            #Immigration births (only happens around pulse time with the influx of births)
            
            mask_em =  (Np[ncs+1:ncs+nf] .> 3) .& (p.K[ncs+1:ncs+nf] .> 1)
            mask_im = (Np[ncs+1:ncs+nf] .== 0) .& (connected_pops[ncs+1:ncs+nf] .> 3)
            extra_b = sum(Births[mask_im] .= 3*μ_p)
            Births[mask_em] .-= extra_b ./ sum(mask_p_s)
            
        end

    end

    v = ones(Int8,tp)
    
    populations  = v*N'+ N*v'
   
    populations[diagind(populations)] = N
    
    out[1:11:end] .= Births
    out[2:11:end] .= S.*Deaths
    out[3:11:end] .= ((beta .* S) ./ populations) * (I + p.ω .* C)
    out[4:11:end] .= E.*Deaths
    out[5:11:end] .= p.ζ .* E
    out[6:11:end] .= p.ρ .* p.γ .* I 
    out[7:11:end] .= I.*Deaths
    out[8:11:end] .= p.γ .* (1 .- p.ρ) .* I
    out[9:11:end] .= R.*Deaths
    out[10:11:end].= (1 ./ Lambda) .* C
    out[11:11:end] .= p.κ .* R 
    
    nothing
end

function ASF_M3S(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β_i, β_o, β_b, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, g, Seasonal, bw, bo, k, la, lo, area, inf_beta,v, nv1,nv2,nv3= p 
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
   
    
    Lambda =  λ + la * cos((t + lo) * 2*pi/year) #decay
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
  
    nv3 .= (matmul!(nv1,v,Nt') .+ matmul!(nv2,reshape(Nt, length(Nt), 1),v')) #calcutlating the populations for combined groups
    
     out[1:11:end] .= Births
     out[2:11:end] .= @. S*Deaths
    out[3:11:end] .= matmul!(inf_beta,((d_eff.*β_o.*S)./nv3),(I.+ω.*C)) .+ β_i .* (S ./ Nt) .* (I .+ ω .* C)
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
    
    Deaths = μ_p*(σ + ((1-σ))* beta_mod) #rate
    Births = p_mag*(σ * Np + ((1-σ)) * sqrt(Np * K))#total! (rate times NP)
    
    
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