module ASF_Model
using LinearAlgebra
using Statistics
using CSV
using DelimitedFiles
using Distributions
using SparseArrays

export asf_model
export density_rate
export frequency_rate
export reparam!

function asf_model_full(out,u,p,t)

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
    μ_bb = copy(p.μ_b)
    μ_dd = copy(p.μ_d)
    Lambda = copy(p.λ)

    for i in 1:Pops.pop #going through populations
    
        nf = Pops.feral[i] #number of feral groups in region
        ncs = Pops.cum_sum[i] #cumsum of farm and ferals over all regions
        N_feral = sum(Np[ncs+1:ncs+nf]) #total feral population in region i
        Density = N_feral/Pops.area[i]

        beta[p.β_d .== i] .*= Density/ref_density

        if p.Seasonal
        
            day = mod(t+offset, year)

            start = p.sd[i]
            len = p.l[i]
            ratio = p.bs[i]

            s_offset = p.so[i]
            s_amp = p.as[i]

            if  start <= day <= start + len
                ratiob = year*ratio/len
                
                μ_bb[ncs+1:ncs+nf] .*= ratiob
                μ_dd[ncs+1:ncs+nf] .*= ratiob
                    
            else
                ratiob = year*(1-ratio)/(year-len)
                
                μ_bb[ncs+1:ncs+nf] .*= ratiob
                μ_dd[ncs+1:ncs+nf] .*= ratiob
            
            end

            Lambda[ncs+1:ncs+nf] .+= s_amp * cos((t + offset .+ s_offset) * 2*pi/year)

        end

    end

    v = ones(Int8,tp)
    
    populations  = v*N'+ N*v'
   
    populations[diagind(populations)] = N

    connected_pops = p.β_b * Np
    
    #Setting base births
    Births = μ_bb.*Np
    
    #Immigration births
    mask_im = (Np .== 0) .& (connected_pops .>1) #population zero but connected groups have 1 or more pigs
    Births[mask_im] .= 2*μ_bb[mask_im]
    total_im = sum(2*μ_bb[mask_im])
    
    #Need to stop boars giving birth!
    mask_boar = (p.μ_c .== 1) .& (Np .> 0) #population greater than 0 but carrying 1
    Births[mask_boar] .= 0 #Dont want these births!
    total_boar = sum(μ_bb[mask_boar].*Np[mask_boar]) #amount of births we have removed
    
    #now we need to adjust for immigration and boar births in the rest of the population
    mask_sow = (Np .> 0) .& (p.μ_c .!= 1) #groups with pigs that are not boars!
    Births[mask_sow] .= Births[mask_sow] .+ (total_boar-total_im)/length(Births[mask_sow])
    
    out[1:11:end] .= Births
    out[2:11:end] .= S.*(μ_dd) .+ dense_deaths(μ_bb, μ_dd, p, S, Np)
    out[3:11:end] .= ((beta .* S) ./ populations) * (I + p.ω .* C)
    out[4:11:end] .= E.*(μ_dd) .+ dense_deaths(μ_bb, μ_dd, p, E, Np)
    out[5:11:end] .= p.ζ .* E
    out[6:11:end] .= p.ρ .* p.γ .* I 
    out[7:11:end] .= I.*(μ_dd) .+ dense_deaths(μ_bb, μ_dd, p, I, Np)
    out[8:11:end] .= p.γ .* (1 .- p.ρ) .* I
    out[9:11:end] .= R.*(μ_dd) .+ dense_deaths(μ_bb, μ_dd, p, R, Np)
    out[10:11:end].= (1 ./ Lambda) .* C
    out[11:11:end] .= p.κ .* R 
    
    nothing
end


function dense_deaths(μ_bb, μ_dd, p, U, N)

    r = μ_bb-μ_dd
    K = p.μ_c
    a = p.g

    dummy = r .* U
    
    ma = N .> K #mask above carrying
    mb = 0 .< N .< K #mask below
    dummy[ma].+= (a[ma].*r[ma]).*(N[ma].-K[ma]).^(1/2).*(U[ma]./N[ma]) #above carrying capacity, extra deaths
    
    dummy[mb] .= -r[mb].*U[mb].*(N[mb]./K[mb]) #below carrying capacity less deaths
    
    return dummy
end



function SEIRC_ODE!(du,u,p,t)
    
    ref_density = 3
    
    u[u.<0].=0 
    
    S = u[1:5:end]
    E = u[2:5:end]
    I = u[3:5:end]
    R = u[4:5:end]
    C = u[5:5:end]
  
    N = S + E + I + R + C .+ 0.0001
    Np = S + E + I + R
    
    Pops = p.Populations 
    
    tp = Pops.cum_sum[end]





    beta = copy(p.β)
    
   
    N_feral = sum(Np) #total feral population 
    Density = N_feral/Pops.area[1]
    beta = beta * Density/ref_density
 

    v = ones(Int8,tp)

    populations  = v*N'+ N*v'
    populations[diagind(populations)] = N;

    connected_pops = p.β_b * Np

    #procceses 
    Births = p.μ_b .* Np
    Births[(p.μ_c .== 1) .& (Np .> 0)] .= 0 #preventing boar populations growing larger than one!
    Births[(Np .== 0) .& (connected_pops .>2)] .= mean(p.μ_b)*2 #allowing migration births if neighbouring groups have pop
    

    
    du[1:5:end] = Births - ((beta.* S) ./ populations) * (I + p.ω .* C) - p.μ_d .* S + (p.μ_b-p.μ_d)./tanh(1).*S.*tanh.(Np./p.μ_c)  + p.κ .* R #S
    du[2:5:end] = ((beta.* S) ./ populations) * (I + p.ω .* C) - p.ζ .* E - p.μ_d .* E + (p.μ_b-p.μ_d)./tanh(1).*E.*tanh.(Np./p.μ_c) #E
    du[3:5:end] = p.ζ .* E - p.γ .* I -  p.μ_d .* I+ (p.μ_b-p.μ_d)./tanh(1).*I.*tanh.(Np./p.μ_c) #I
    du[4:5:end] = p.γ .* (1 .- p.ρ) .* I - p.μ_d .* R + (p.μ_b-p.μ_d)./tanh(1).*R.*tanh.(Np./p.μ_c) - p.κ .* R  #R
    du[5:5:end] = p.ρ .* p.γ .* I + p.μ_d .* I+ (p.μ_b-p.μ_d)./tanh(1).*I.*tanh.(Np./p.μ_c) - p.λ .* C#C
    nothing
end

function reparam!(input)

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