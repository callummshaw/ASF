module ASF_Model
using LinearAlgebra
using Statistics
using JLD2
using CSV
using DelimitedFiles
using GraphPlot,Compose,Graphs,Colors,Cairo
using Distributions

export density_rate
export analyse_out
export frequency_rate
export save_output
export plot_network
export reparam!

function density_rate(out,u,p,t)
   
    u[u.<0].=0 
    S = u[1:5:end]
    E = u[2:5:end]
    I = u[3:5:end]
    R = u[4:5:end]
    C = u[5:5:end]
    
    N = S + E + I + R + C .+ 0.001
    Np = S + E + I + R
    
    Pops = p.Populations 
    
    tp = Pops.cum_sum[end]
    beta = copy(p.β)
    
    for i in 1:Pops.pop
        j = i + 1 

        nf = Pops.feral[i] #number of feral groups in region
        nt = Pops.total[i] #number of feral groups and farms in region
        ncs = Pops.cum_sum[i] #cumsum of farm and ferals over all regions

        N_feral = sum(N[ncs+1:ncs+nf]) #total feral population in region i
        Density = N_feral/Pops.area[i]
        beta[p.β_d .== j] .*= Density
    end

    column(i) = N .+ N[i]
    populations  = hcat([column(i) for i=1:tp]...)
    populations[diagind(populations)] = N;
    
    Births = p.μ_b.*Np #+ 0.001*(p.μ_b .* (p.β_b * Np))
    Infect = ((beta .* S) ./ populations) * (I + p.ω .* C)#ASF Infections
    Infectous = p.ζ .* E
    Recover = p.γ .* (1 .- p.ρ) .* I #ASF Recoveries
    Death_I = p.ρ .* p.γ .* I #ASF Deaths in I
    Death_nI = p.μ_d .* I+ p.μ_g .* (Np .* I) #Natural Deaths in I
    Death_S = p.μ_d .* S + p.μ_g .* (Np .* S) #Natural Deaths S
    Death_E = p.μ_d .* E + p.μ_g .* (Np .* E)
    Death_R = p.μ_d .* R + p.μ_g .* (Np .* R) #Natural Deaths R
    Decay_C = p.λ .* C #Body Decomposition 
    
    
    out[1:10:end] = Births
    out[2:10:end] = Death_S
    out[3:10:end] = Infect
    out[4:10:end] = Death_E
    out[5:10:end] = Infectous
    out[6:10:end] = Death_I
    out[7:10:end] = Death_nI
    out[8:10:end] = Recover
    out[9:10:end] = Death_R
    out[10:10:end] = Decay_C
end

function frequency_rate(out,u,p,t)
   
    u[u.<0].=0 
    S = u[1:5:end]
    E = u[2:5:end]
    I = u[3:5:end]
    R = u[4:5:end]
    C = u[5:5:end]
    
    N = S + E + I + R + C .+ 0.001
    Np = S + E + I + R
    
    Pops = p.Populations 
    
    tp = Pops.cum_sum[end]
    beta = copy(p.β)
    
    reference_density = 4
    
    for i in 1:Pops.pop
        j = i + 1 

        nf = Pops.feral[i] #number of feral groups in region
        nt = Pops.total[i] #number of feral groups and farms in region
        ncs = Pops.cum_sum[i] #cumsum of farm and ferals over all regions

        N_feral = sum(N[ncs+1:ncs+nf]) #total feral population in region i
        beta[p.β_d .== j] *= reference_density

    end

    column(i) = N .+ N[i]
    populations  = hcat([column(i) for i=1:tp]...)
    populations[diagind(populations)] = N;
    
    Births = p.μ_b.*Np #+ 0.001*(p.μ_b .* (p.β_b * Np))
    Infect = ((beta .* S) ./ populations) * (I + p.ω .* C)#ASF Infections
    Infectous = p.ζ .* E
    Recover = p.γ .* (1 .- p.ρ) .* I #ASF Recoveries
    Death_I = p.ρ .* p.γ .* I #ASF Deaths in I
    Death_nI = p.μ_d .* I+ p.μ_g .* (Np .* I) #Natural Deaths in I
    Death_S = p.μ_d .* S + p.μ_g .* (Np .* S) #Natural Deaths S
    Death_E = p.μ_d .* E + p.μ_g .* (Np .* E)
    Death_R = p.μ_d .* R + p.μ_g .* (Np .* R) #Natural Deaths R
    Decay_C = p.λ .* C #Body Decomposition 
    
    
    out[1:10:end] = Births
    out[2:10:end] = Death_S
    out[3:10:end] = Infect
    out[4:10:end] = Death_E
    out[5:10:end] = Infectous
    out[6:10:end] = Death_I
    out[7:10:end] = Death_nI
    out[8:10:end] = Recover
    out[9:10:end] = Death_R
    out[10:10:end] = Decay_C
end

function reparam!(input)

    rebeta!(input) #keeping same links but assinging new values
    
    init_pops = input.U0
    pops = input.Populations_data
    counts = input.Parameters.Populations
    
    K = init_pops[1:5:end] + init_pops[2:5:end] + init_pops[3:5:end] #carrying capacity of each group
    
    # All other params
    
    ζ = [] #latent rate
    γ = [] #recovery/death rate
    μ_b = [] #births
    μ_d = [] #natural death rate
    μ_g = [] #density dependent deaths
    ω = [] #corpse infection modifier
    ρ = [] #ASF mortality
    λ = [] #corpse decay rate
    
    for i in 1:counts.pop
        data =  pops[i]
        
        nf = counts.feral[i]
        nl = counts.farm[i]
        nt = counts.total[i]
        
        cs = counts.cum_sum
        
        ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2],0,5) #latent dist
        γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2],0,5) #r/d rate dist
        μ_b_d = TruncatedNormal(data.Birth[1], data.Birth[2],0,1) #birth dist
        μ_d_d = TruncatedNormal(data.Death_n[1], data.Death_n[2],0,1) #n death dist
        ω_d = TruncatedNormal(data.Corpse[1], data.Corpse[2],0,1) #corpse inf dist
        ρ_d = TruncatedNormal(data.Death[1], data.Death[2],0,1) #mortality dist
        λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2],0,1) #corpse decay feral dist
        λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2],0,5) #corpse decay farm dist

        append!(ζ,rand(ζ_d,nt))
        append!(γ,rand(γ_d,nt))
        append!(ω,rand(ω_d,nt))
        append!(ρ,rand(ρ_d,nt))

        μ_b_r = rand(μ_b_d,nt)
        μ_d_r = rand(μ_d_d,nt)

        append!(μ_b, μ_b_r)
        append!(μ_d, μ_d_r) 

        μ_g_r =  (μ_b_r-μ_d_r)./K[cs[i]+1:cs[i+1]]
        append!(μ_g,μ_g_r)

        append!(λ,rand(λ_fd,nf))
        append!(λ,rand(λ_ld,nl))
        
    end
    
    
    
    input.Parameters.γ = γ
    input.Parameters.ζ = ζ
    input.Parameters.λ = λ
    input.Parameters.μ_b = μ_b 
    input.Parameters.μ_d = μ_d
    input.Parameters.μ_g = μ_g
    input.Parameters.ω = ω 
    input.Parameters.ρ = ρ
    
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


function analyse_out(input, counts)
    
    classes, t_steps, n_ens = size(input) 
    
    nft_alive = Vector{Vector{Float64}}(undef, n_ens)
    nft_exposed = Vector{Vector{Float64}}(undef, n_ens)
    
    
    nlt_alive = Vector{Vector{Float64}}(undef, n_ens)
    nlt_exposed = Vector{Vector{Float64}}(undef, n_ens)
    
    for i in 1:n_ens
        
        sol = input[i]
        
        data = reduce(vcat,transpose.(sol.u))

        if any(x->x==-1, data) == true
            println("Need to Reduce Timestep")
            
            #break
        end

        s_d = data[:,1:5:end]
        e_d = data[:,2:5:end]
        i_d = data[:,3:5:end]
        r_d = data[:,4:5:end]
        c_d = data[:,5:5:end]

        np = counts.pop

        nf_alive = Vector{Float64}(undef, np)
        nf_exposed = Vector{Float64}(undef, np)

        pl_alive = Vector{Float64}(undef, np)
        pl_exposed = Vector{Float64}(undef, np)


        for j in 1:np

            #FERAL
            
            ll = counts.cum_sum[j]+1
            uu = counts.cum_sum[j] + counts.feral[j]

            #disease-free classes
            s_j = s_d[:,ll:uu]
            r_j = r_d[:,ll:uu]

            final = s_j[end,:]+r_j[end,:]
            na = count(x->x!=0,final)

            nf_alive[j] = 100* na/counts.feral[j]

             #disease classes
            e_j = e_d[:,ll:uu]
            i_j = i_d[:,ll:uu]
            c_j = c_d[:,ll:uu]

            dis = e_j + i_j +c_j 
            ndis = sum(dis,dims=1)
            ne = count(x->x!=0,ndis)
            nf_exposed[j] = 100*ne/counts.feral[j]
            
            #FARM
            
            l_l = uu +1 
            u_l = counts.cum_sum[j+1]

            #disease-free classes
            s_j = s_d[:,l_l:u_l]
            r_j = r_d[:,l_l:u_l]

            final = s_j[end,:]+r_j[end,:]
            na = count(x->x!=0,final)

            pl_alive[j] = 100* na/counts.farm[j]

             #disease classes
            e_j = e_d[:,l_l:u_l]
            i_j = i_d[:,l_l:u_l]
            c_j = c_d[:,l_l:u_l]

            dis = e_j + i_j +c_j 
            ndis = sum(dis,dims=1)
            ne = count(x->x!=0,ndis)
            pl_exposed[j] = 100*ne/counts.farm[j]

        end
        
        nft_alive[i] = nf_alive
        nft_exposed[i] = nf_exposed
        
        nlt_alive[i] = pl_alive
        nlt_exposed[i] = pl_exposed
        
    end
    
    println("------------------------------------------------")
    println("Run Stats:")
    println(string(counts.pop, " Populations"))
    println(string(n_ens, " Runs"))
    println("------------------------------------------------")
    println("% of groups exposed to ASF in each population:")
    println(mean(nft_exposed))
    println("------------------------------------------------")
    println("% of groups that survive in each population:")
    println(mean(nft_alive))
    println("------------------------------------------------")
    if sum(counts.farm) > 0 
        println("% of farms exposed to ASF in each population:")
        println(mean(nlt_exposed))
        println("------------------------------------------------")

        println("% of farms that survive in each population:")
        println(mean(nlt_alive))
    end 
    
    #return mean(nft_alive), mean(nft_exposed), mean(nlt_alive), mean(nlt_exposed)

end

function save_output(output, data, path, name)
    #=
    Function to save output
    =#
    
    dims = length(size(output))
    
    if dims == 3 #ensemble run
        classes, t_steps, n_ens = size(output)
    else #single run
        classes, t_steps = size(output)
        n_ens = 1
    end
    
    #making save directory
    dir = string(path, "Results/", name)
    
    isdir(dir) || mkdir(dir)
    cd(dir)

    #copying inputs
    cp(string(path,"Inputs"), "Inputs",force = true)
    
    #saving run parameters
    save_object("parameters.jdl2", data)
    
    #saving output
    if dims == 3 #ensemble
        for i = 1:n_ens
            solnew = vcat(output.u[i].t',output[:,:,i])' #data
            writedlm( string("ASF_",i,".csv"),  solnew, ',')
        end
    else
        solnew = vcat(sol.t',sol[:,:,i])'
        writedlm( "ASF_single",  solnew, ',')
    end
    
    
end

function summuary__stats(output, counts)
    
    classes, t_steps, n_ens = size(output) 
    
    
    nft = Array{Int64}(undef, n_ens, 2)
    
    for i in 1:n_ens
        
        sol = output[i]
        
        data = reduce(vcat,transpose.(sol.u))

        if any(x->x==-1, data) == true
            println("Need to Reduce Timestep")
            
            #break
        end

        s_d = data[:,1:5:end]
        e_d = data[:,2:5:end]
        i_d = data[:,3:5:end]
        r_d = data[:,4:5:end]
        c_d = data[:,5:5:end]

        np = counts.pop

        for j in 1:np #we only have 1 pop

            #FERAL
            
            ll = counts.cum_sum[j]+1
            uu = counts.cum_sum[j] + counts.feral[j]

            #disease-free classes
            s_j = s_d[:,ll:uu]
            r_j = r_d[:,ll:uu]

            final = s_j[end,:]+r_j[end,:]
            na = count(x->x!=0,final)


             #disease classes
            e_j = e_d[:,ll:uu]
            i_j = i_d[:,ll:uu]
            c_j = c_d[:,ll:uu]

            dis = e_j + i_j +c_j 
            ndis = sum(dis,dims=1)
            ne = count(x->x!=0,ndis)
            
            nft[i,1] = na
            nft[i,2] = ne
        end
        
    end
    
    return nft

end
function plot_network(input, output)
    connections = input.Parameters.β_b
    gg = Graphs.SimpleGraph(connections)

    
    dims = length(size(output))
    
    if dims == 3 #ensemble run
        classes, t_steps, n_ens = size(output)
    else #single run
        classes, t_steps = size(output)
        n_ens = 1
    end

    storage = zeros(t_steps, div(classes,5))

    for i = 1:n_ens

        data  = vcat(output[:,:,i])'
        E = data[:,2:5:end];
        I = data[:,3:5:end];
        C = data[:,5:5:end];

        NI = E + I + C

        NI[NI .> 0 ] .= 1

        storage += NI
    end
    
    
    node_status = floor.(Int,storage /20) .+ 2
    node_status[storage .== 0] .= 1
    
    nodecolor = [colorant"#FCE1A4",colorant"#FABF7B",colorant"#F08F6E",colorant"#E05C5C",colorant"#D12959", colorant"#AB1866",colorant"#6E005F"]
    
    # membership color
    #1- 0% FCE1A4 
    #2- 1-19% FABF7B
    #3- 20-39% F08F6E
    #4- 40-59% E05C5C
    #5- 60-79% D12959
    #6- 80-99% AB1866
    #7- 100% 6E005F
    
    #where to save
    cd("/home/callum/ASF/Results/Plots/")
    
    #assuming 10 time steps per day, therefor 300 per month, will plot monthly
    period = 300
    n_months = div(t_steps, period)
    
    for i = 1:n_months
        j = period*(i-1)+1
        nodefillc = nodecolor[node_status[j,:]]
        draw(PNG(string("Sample_",i,".png"), 20cm, 20cm), gplot(gg, nodefillc=nodefillc, layout=circular_layout))
    end
    
end

end