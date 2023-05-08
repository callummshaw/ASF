module Models 

using LinearAlgebra
using Distributions
using SparseArrays

export ASF_M3
export ASF_M3S
export ASF_M2
export ASF_M1



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

    β_i, β_o, β_b, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, Seasonal, bw, bo, k, la, lo, Area = p 
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

    KT = sum(K)

    
    beta_mod = abs.(tp/KT).^η
    
    if Seasonal    
        Lambda = λ + la * cos.((t + lo) * 2*pi/year) #decay
        p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
    else 
        Lambda = λ
        p_mag = μ_p
    end

    if θ == 0.5 #density births and death!
        Deaths = μ_p.*(σ .+ ((1-σ)).*sqrt.(abs.(Np./K)))*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* sqrt.(abs.(Np .* K)))#total! (rate times NP)
    else
        Deaths = μ_p.*(σ .+ ((1-σ)).*(Np./K).^θ)*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* Np.^(1-θ) .* K.^θ)#total! (rate times NP)
    end

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

function ASF_M2(out,u,p,t)
    #ASF model for a single population (can make some speed increases) without farms!

    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, Seasonal, bw, bo, k, la, lo, Area = p 
    
    u[u.<0] .= 0
    
    S, E, I, R, C = u
    N = sum(u)
    Np = S + E + I + R
    
    if N == 0 
        N = 1
    end
   
    beta_mod = abs.(tp/KT).^η
    
    if Seasonal    
        Lambda = λ + la * cos.((t + lo) * 2*pi/year) #decay
        p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
    else 
        Lambda = λ
        p_mag = μ_p
    end

    if θ == 0.5 #density births and death!
        Deaths = μ_p.*(σ .+ ((1-σ)).*sqrt.(abs.(Np./K)))*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* sqrt.(abs.(Np .* K)))#total! (rate times NP)
    else
        Deaths = μ_p.*(σ .+ ((1-σ)).*(Np./K).^θ) #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* Np.^(1-θ) .* K.^θ)#total! (rate times NP)
    end
    
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
    out[10] = (1 / Lambda) * C
    out[11] = κ * R 

    nothing
end


function ASF_M1(du,u,p,t)
    
    #ode equivelent of our ASF model
    β, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, Seasonal, bw, bo, k, la, lo, Area = p 
    
    S, E, I, R, C = u
    N = sum(u)
    Np = S + E + I + R
    
    beta_mod = abs.(tp/KT).^η
    
    if Seasonal    
        Lambda = λ + la * cos.((t + lo) * 2*pi/year) #decay
        p_mag = birth_pulse_vector(t,k,bw,bo) #birth pulse value at time t
    else 
        Lambda = λ
        p_mag = μ_p
    end

    if θ == 0.5 #density births and death!
        ds = μ_p.*(σ .+ ((1-σ)).*sqrt.(abs.(Np./K)))*g #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* sqrt.(abs.(Np .* K)))#total! (rate times NP)
    else
        ds = μ_p.*(σ .+ ((1-σ)).*(Np./K).^θ) #rate
        Births = p_mag.*(σ .* Np .+ ((1-σ)) .* Np.^(1-θ) .* K.^θ)#total! (rate times NP)
    end
    
    du[1] = Births + κ*R - ds*S - beta_mod*β*(I + ω*C)*S/N
    du[2] = beta_mod*β*(I + ω*C)*S/N - (ds + ζ)*E
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

function reparam!(input)
    #needs to be updated for new model input!
    rebeta!(input) #keeping same links but assinging new values
    
    init_pops = input.U0
    pops = input.Populations_data
    counts = input.Parameters.Populations
    
    birth_death_mod = 0.8
   
    K = init_pops[1:5:end] + init_pops[2:5:end] + init_pops[3:5:end] #carrying capacity of each group
    
    cs = counts.cum_sum

    # All other params
    n_groups = length(K)
    ζ = Vector{Float32}(undef, n_groups) #latent rate
    γ = Vector{Float32}(undef, n_groups) #recovery/death rate
    μ_b = Vector{Float32}(undef, n_groups) #births
    μ_d = Vector{Float32}(undef, n_groups) #natural death rate
    μ_c = Vector{UInt8}(undef, n_groups) #density dependent deaths
    ω = Vector{Float32}(undef, n_groups) #corpse infection modifier
    ρ = Vector{Float32}(undef, n_groups) #ASF mortality
    λ = Vector{Float32}(undef, n_groups) #corpse decay rate
    κ = Vector{Float32}(undef, n_groups)

    for i in 1:counts.pop
        data =  pops[i]
        
        nf = counts.feral[i]
        nl = counts.farm[i]
        nt = counts.total[i]
        
        ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2], 0, 5) #latent dist
        γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2], 0, 5) #r/d rate dist
        μ_b_d = TruncatedNormal(data.Birth[1], data.Birth[2], 0, 1) #birth dist
        #μ_d_d = TruncatedNormal(data.Death_n[1], data.Death_n[2], 0, 1) #n death dist
        ω_d = TruncatedNormal(data.Corpse[1], data.Corpse[2], 0, 1) #corpse inf dist
        ρ_d = TruncatedNormal(data.Death[1], data.Death[2], 0, 1) #mortality dist
        λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2], 0, 1) #corpse decay feral dist
        λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2], 0, 5) #corpse decay farm dist
        κ_d = TruncatedNormal(data.Immunity[1], data.Immunity[2], 0, 1)

        ζ[cs[i]+1:cs[i+1]] = rand(ζ_d,nt)
        γ[cs[i]+1:cs[i+1]] = rand(γ_d,nt)
        ω[cs[i]+1:cs[i+1]] = rand(ω_d,nt)
        ρ[cs[i]+1:cs[i+1]] = rand(ρ_d,nt)
        κ[cs[i]+1:cs[i+1]] = rand(κ_d,nt)

        μ_b_r = rand(μ_b_d,nt) 
        μ_b[cs[i]+1:cs[i+1]] = μ_b_r
        μ_d[cs[i]+1:cs[i+1]] = birth_death_mod*μ_b_r
        μ_c[cs[i]+1:cs[i+1]] = K[cs[i]+1:cs[i+1]]

        λ[cs[i]+1:cs[i]+nf] = rand(λ_fd,nf)
        λ[cs[i]+nf+1:cs[i+1]] = rand(λ_ld,nl)

        
    end
    
    input.Parameters.γ = γ
    input.Parameters.ζ = ζ
    input.Parameters.λ = λ
    input.Parameters.μ_b = μ_b 
    input.Parameters.μ_d = μ_d
    input.Parameters.μ_c = μ_c
    input.Parameters.ω = ω 
    input.Parameters.ρ = ρ
    input.Parameters.κ = κ
    
end

function rebeta!(input)
    
    pops = input.Populations_data
    counts = input.Parameters.Populations
    
    beta = copy(input.Parameters.β)
    
    for i in 1:counts.pop

            data = pops[i]
            nf = counts.feral[i]
            nt = counts.total[i]
            
            beta_sub = beta[counts.cum_sum[i]+1:counts.cum_sum[i+1],counts.cum_sum[i]+1:counts.cum_sum[i+1]]
            
            n_aim = data.N_int[1]
        
            i_f = TruncatedNormal(data.B_f[1],data.B_f[2],0,5) #intra group
            i_ff = TruncatedNormal(data.B_ff[1],data.B_ff[2],0,5) #inter group

            for k in 1:nf
                for j in 1:nf
                    if k == j
                        beta_sub[j,j] = rand(i_f)
                    elseif k>j && beta_sub[k,j] != 0
                        beta_sub[k,j] = beta_sub[j,k]  = rand(i_ff) .* (1/n_aim) 
                    end
                end
            end

            if counts.farm[i] > 0

                i_l = TruncatedNormal(data.B_l[1],data.B_l[2],0,5)
                i_fl = TruncatedNormal(data.B_fl[1],data.B_fl[2],0,5)

                for k in 1:nt
                    for j in nf+1:nt
                        if k == j
                            beta_sub[j,j] = rand(i_l)
                        elseif j>k && beta_sub[k,j] != 0
                            beta_sub[k,j] = beta_sub[j,k] = rand(i_fl)
                        end
                    end 
                end

            end 

            beta[counts.cum_sum[i]+1:counts.cum_sum[i+1],counts.cum_sum[i]+1:counts.cum_sum[i+1]] = beta_sub
    end
    
    input.Parameters.β = beta
    
end

end