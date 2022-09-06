module ASF_Model
using LinearAlgebra
using Statistics
using CSV
using DelimitedFiles
using Distributions

export density_rate
export frequency_rate
export reparam!


function density_rate(out,u,p,t)
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

    for i in 1:Pops.pop
    
        nf = Pops.feral[i] #number of feral groups in region
        ncs = Pops.cum_sum[i] #cumsum of farm and ferals over all regions
        N_feral = sum(Np[ncs+1:ncs+nf]) #total feral population in region i
        Density = N_feral/Pops.area[i]

        beta[p.β_d .== i] .*= Density/ref_density
    end

    column(i) = N .+ N[i]
    populations  = hcat([column(i) for i=1:tp]...)
    populations[diagind(populations)] = N;

  
    connected_pops = p.β_b * Np
 
    Births = p.μ_b .* Np
    Births[(p.μ_c .== 1) .& (Np .> 0)] .= 0 #preventing boar populations growing larger than one!
    Births[(Np .== 0) .& (connected_pops .>2)] .= mean(p.μ_b)*2 #allowing migration births if neighbouring groups have pop
    Infect = ((beta .* S) ./ populations) * (I + p.ω .* C)#ASF Infections
    Infectous = p.ζ .* E
    Recover = p.γ .* (1 .- p.ρ) .* I #ASF Recoveries
    Death_I = p.ρ .* p.γ .* I #ASF Deaths in I
    Death_nI = p.μ_d .* I+ (p.μ_b-p.μ_d)./tanh(1).*I.*tanh.(Np./p.μ_c) #Natural Deaths in I
    Death_S = p.μ_d .* S + (p.μ_b-p.μ_d)./tanh(1).*S.*tanh.(Np./p.μ_c) #Natural Deaths S
    Death_E = p.μ_d .* E + (p.μ_b-p.μ_d)./tanh(1).*E.*tanh.(Np./p.μ_c)
    Death_R = p.μ_d .* R + (p.μ_b-p.μ_d)./tanh(1).*R.*tanh.(Np./p.μ_c) #Natural Deaths R
    Decay_C = p.λ .* C #Body Decomposition 
    W_Immunity = p.κ .* R .* 0
   
    out[1:11:end] = Births
    out[2:11:end] = Death_S
    out[3:11:end] = Infect
    out[4:11:end] = Death_E
    out[5:11:end] = Infectous
    out[6:11:end] = Death_I
    out[7:11:end] = Death_nI
    out[8:11:end] = Recover
    out[9:11:end] = Death_R
    out[10:11:end] = Decay_C
    out[11:11:end] = W_Immunity
    
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
    ζ = Vector{Float64}(undef, n_groups) #latent rate
    γ = Vector{Float64}(undef, n_groups) #recovery/death rate
    μ_b = Vector{Float64}(undef, n_groups) #births
    μ_d = Vector{Float64}(undef, n_groups) #natural death rate
    μ_c = Vector{Int32}(undef, n_groups) #density dependent deaths
    ω = Vector{Float64}(undef, n_groups) #corpse infection modifier
    ρ = Vector{Float64}(undef, n_groups) #ASF mortality
    λ = Vector{Float64}(undef, n_groups) #corpse decay rate
    κ = Vector{Float64}(undef, n_groups)

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