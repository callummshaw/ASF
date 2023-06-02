"Module that builds most of the ASF model input"

module Input

using LinearAlgebra
using Distributions
using Graphs
using QuadGK
using CSV
using DataFrames

include("Population.jl")
include("Beta.jl")
include("Network.jl")

export Model_Data

const year = 365 
abstract type Data_Input end

struct Meta_Data <: Data_Input
    #=
    Structure to store all the meta data infomation for the run:
    number of years, if the model is being run in ensemble, if
    using distros or just mean values for parameters, the number of 
    populations and the number of said populations seeded with ASF
    =#
    Model::Int8 #which model we are running with 1-ODE 2-TauHomogeneous 3-TauHeterogeneous
    Fitted::Bool #is model run with params from previous fitting?
    years::Float32 #years simulation will run for
    S_day::Int16 #starting date of simulation
    N_ensemble::Int64 #number of runs in an ensemble
    Identical::Bool #if we params to be drawn from dist or just means of dist
    Seasonal::Bool #If model is seasonal
    Network::String #Type of network used for model random (r), scale-free (s), or small worlds (w)
    N_param::Float32 #Network parameter (only for small worlds)
    N_Pop::Int8 #number of populations we want!
    N_Seed::Int8 #population seeded with ASF
    N_Con::Int8 #Number of connecting groups between populations

    function Meta_Data(input, verbose)
       
        Mn = parse(Int8, input.Value[1]) #model
        F = (input.Value[2] == "true") #Fitted
        Ny = parse(Int16, input.Value[3])#years
        Sd = parse(Int16, input.Value[4])
        Ne = parse(Int64,input.Value[5])# number of runs 
        I  = (input.Value[6] == "true") #indetical
        S  = (input.Value[7] == "true") #seasonal
        Nw = input.Value[8]
        Nps= parse(Float32, input.Value[9])

        Np = parse(Int8, input.Value[10]) #number of populations
        Ns = parse(Int8, input.Value[11]) #population seeded with ASF
        NC = parse(Int8, input.Value[12]) #average degree of population

        if verbose
            if Mn == 1
                @info "ODE Model"
            elseif Mn == 2
                @info "Tau Leaping Homogeneous Model"
            elseif Mn == 3
                @info "Tau Leaping Heterogeneous Model"
                
                if Nw == "s" #only works with even degrees            
                    @info "Barabasi Albert Scale Free Network"
                elseif Nw == "w" #only works with even degrees
                    @info "Watts Strogatz Small Worlds Network"
                    @info "Rho: $(Nps)"
                else
                    @info "Erdos Renyi Random Network"
                end

            end
            
            @info "$(Ne) Simulations"
                
            if F
                @info "Running with fitted parameters"
            end

            if S
                @info "Running with seasons"
            end 

              
            if Ny > 10
                @warn "Running for $(Ny) years"
            end

            if Ne > 1000
                @warn "Running with ensemble size of $(Ne)"
            end

            if Ns > Np
                @warn "Seeding ASF in population outside of population limit"
                Ns = copy(Np)
            end
        end

        new(Mn,F,Ny,Sd,Ne,I,S,Nw, Nps, Np,Ns,NC,PNT)
    
        
    end
end

struct Population_Data <: Data_Input
    #=
    Structure to store parameters for each population
    =#

    Dense::Vector{Float64} #density of population
    N_feral::Vector{UInt16} #number of feral groups 
    N_f::Vector{Float64} #feral group size
    Boar_p::Vector{Float64} #Proportion of groups that are wild boars
    N_int::Vector{UInt8} #average interconnection between feral groups
    N_e::Vector{Float64} #% of population seeded in exposed
    N_i::Vector{Float64} #% of population seeded in infected
    
    LN::Vector{Float64} #number of litters per year
    LS::Vector{Float64} #litter size
    LM::Vector{Float64} #first year mortality rate
   
    Density_rate::Vector{Float32} #percent of natural birth/deaths that are NOT related to population density (0-1)
    Density_power::Vector{Float32} #power of K/N or N/K use for density birth/death rates
    g_fit::Vector{Float32} #fitting param to ensure stable birth/death rate
   
    B_f::Vector{Float64} #intra feral group transmission
    B_ff::Vector{Float64} #inter feral group transmission
    Corpse::Vector{Float64} #corpse infection modifier
    
    Death::Vector{Float64} #ASF death prob
    Latent::Vector{Float64} #latent period
    Recovery::Vector{Float64}#Recovery rate
    Immunity::Vector{Float64} #immunity
    Decay_f::Vector{Float64} #decay feral
   
    #farm
    N_farm::Vector{UInt8} #number of farm groups
    N_l::Vector{Float64} #farm population size
    B_l::Vector{Float64} #intra farm transmission
    B_fl::Vector{Float64} #farm-feral transmission
    Decay_l::Vector{Float64} #decay farm
    
    function Population_Data(input, sim, verbose)
        
        #Feral/General Params
        
        Den = [input.Mean[1],input.STD[1]] #density
        Nf = [input.Mean[2],input.STD[2]] #number feral
        Sf = [input.Mean[3],input.STD[3]] #feral group size
        Br = [input.Mean[4],input.STD[4]] #boar percentage
        Ni = [input.Mean[5],input.STD[5]] #inter feral conectivity
        N_e = [input.Mean[6],input.STD[6]] #percentage of pop exposed
        N_i = [input.Mean[7],input.STD[7]] #percentage of pop infected
        
        LN = [input.Mean[8],input.STD[8]] #number of litters
        LS = [input.Mean[9],input.STD[9]] #litter size
        LM = [input.Mean[10],input.STD[10]] #first year mortality
        
        Dr = [input.Mean[11],input.STD[11]] #density rate
        Dp = [input.Mean[12],input.STD[12]] #density power
        Gf = [input.Mean[13],input.STD[13]] #param to stop pop drift
        
       
        #fitted params, will run off these values if not fitted!
        Bf = [input.Mean[14],input.STD[14]] #intra feral
        Bff = [input.Mean[15],input.STD[15]] #inter feral
        C = [input.Mean[16],input.STD[16]] #corpse infection modifier
        
        D = [input.Mean[17],input.STD[17]] #death prob once infected 
        
        L = day_to_rate(input.Mean[18],input.STD[18]) #latent rate
        R = day_to_rate(input.Mean[19],input.STD[19]) #recovery rate
        Im = day_to_rate(input.Mean[20],input.STD[20]) #immunity rate
        Df = [input.Mean[21],input.STD[21]] #corpse decay period
        
        #Farm Params
        
        Nl = [input.Mean[22],input.STD[22]] #number of farms in pop
        Sl = [input.Mean[23],input.STD[23]] #farm size
        Bl = [input.Mean[24],input.STD[24]]
        Bfl = [input.Mean[25],input.STD[25]]
        Dl = [input.Mean[26],input.STD[26]]
                    
        #Checking the inputs to make sure they are reasonable/expected
        if verbose 
            if Den[1] > 10
                @warn "High starting density of $(Den[1])"
            end
            if sim.Model == 3
                @info "Boar to group ratio of $(Br[1])"
                @info "Mean feral connectivity of $(Ni[1])"
                if (Nf[1] > 1000) | (Nf[1] < 100)
                    @warn "$(Nf[1]) feral groups"
                end
            else 
                
            end 
            
            if (Sf[1] > 25) | (Sf[1]<4)
                @warn "Mean feral group size of $(Sf[1])"
            end
            
            mean_births = 0.5*LN[1]*LS[1]*(1-LM[1])/365
            
            if (mean_births > 0.01) | (mean_births < 0.001)
                @warn "Birth rate of $(mean_births)"
            end  

            if (Dr[1] > 1 ) | (Dr[1] < 0)
                @warn "Density_rate must be between 0-1"
            end 


            if D[1] != 0.95
                @warn "Death probability of $(D[1])"
            end 

            if (Bf[1] > 1)
                @warn "Intra-group transmission of $(Bf[1])"
            end 

            if (Bff[1] > 1)
                @warn "Inter-group transmission of $(Bff[1])"
            end 

        end
     
        new(Den, Nf, Sf, Br, Ni, N_e, N_i, LN, LS, LM, Dr, Dp, Gf, Bf, Bff, C, D, L, R, Im, Df, Nl, Sl, Bl, Bfl, Dl)
    end
    
end

struct Seasonal_effect
    #= 
    Structure to store key seasonal data
    =#
    
    #Birth/Death effects use a birth pulse, constant deaths over the year
    Birth_width::Float32 #width of birth pulse
    Birth_offset::Float32 #offset of birth pulse (at 0 centred at 182.5)

    #Decay effects sine function 
    Decay_amp::Float32 #amp of function
    Decay_offset::Float32 #offset of decay, running with cosine so maximum at start of year (European...)

    function Seasonal_effect(input,pop_data,verbose)
        
        Bw = input.Value[1]
        Bo = input.Value[2]
        
        Da = input.Value[3]
        Do = input.Value[4]


        if verbose
            if (Bw > 20) 
                @warn "Very narrow birth pulse"
            end

            if Da > 35
                @warn "Large yearly range in decay times"
            end
        
        end

    new(Bw, Bo, Da, Do)           

    end

end


mutable struct Model_Parameters
    #=
    Structure to store key parameters
    =#
    β::Matrix{Float32} #transmission matrix
    β_b::Matrix{Int16} #what feral groups are linked to each other, for births
    β_d::Matrix{Int16} #used to identify feral groups and farms for each population for density calcualtions

    μ_p::Vector{Float32} #birth/death rate at K
    K::Vector{Int16} #carrying capicity
   
    ζ::Vector{Float32} #latent rate
    γ::Vector{Float32} #recovery rate
    ω::Vector{Float32} #corpse infection modifier
    ρ::Vector{Float32} #death probability
    λ::Vector{Float32} #corpse decay rate
    κ::Vector{Float32} #loss of immunity rate
    
    σ::Vector{Float32} #density split
    θ::Vector{Float32} #density power for births/deaths
    g::Vector{Float32} #fitting param for stable populations

    Seasonal::Bool #if we are running with seasonality

    bw::Vector{Float32}
    bo::Vector{UInt16} 
    k::Vector{Float32} 
    la::Vector{Float32}
    lo::Vector{UInt16}

    Populations::Network.Network_Data #breakdown of population
    
    function Model_Parameters(sim, pops, sea, U0, Populations, network)
        
        β, connected_pops, connected_births = Beta.construction(sim, pops, Populations, network)
        μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, g, bw, bo, k, la, lo  = parameter_build(sim, pops, sea, U0, Populations)
        
        new(β, connected_births, connected_pops, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, g, sim.Seasonal, bw, bo, k, la, lo, Populations)
    end
    
end

struct Model_Data
    #=
    Structure to store key data on mode
    =#
    MN::Int8 #which model we are running with (1- ODE, 2-Tau Homogeneous, 3-Tau Heterogeneous)
    Time::Tuple{Float32, Float32} #Model run time
    NR::Int64 #number of runs in ensemble!
    U0::Vector{Int32} #Initial Population
    Parameters::Model_Parameters #Model parameters
    Populations_data::Vector{Population_Data} #distributions for params

    function Model_Data(Path; pop_net= 0, verbose = false)
        #custom_network parameter used for fitting!
        sim, pops, sea = read_inputs(Path, verbose)

        Time = (sim.S_day,sim.years*365+sim.S_day)

        #now building feral pig network
        network, counts = Network.build(sim, pops, verbose, pop_net) 
        
        #Now using network to build init pops
        S0 = Population.build_s(sim, pops, network, counts, verbose) #initial populations
     
        Parameters = Model_Parameters(sim, pops, sea, S0, counts, network)
    
        U0 = Population.spinup_and_seed(sim,pops, S0,Parameters, verbose) #burn in pop to desired start day and seed desired ASF!
        new(sim.Model,Time, sim.N_ensemble, U0, Parameters, pops)
        
    end
    
end


function day_to_rate(Mean, STD)
    #simple conversion to switch between time and rate
    mean_rate = 1/Mean
    std_rate = 1/Mean - 1/(Mean+STD)

    return [mean_rate, std_rate]
end

function parameter_build(sim, pops, sea, init_pops, counts)
    #=
    Function that builds most parameters for model
    =#
    
    K = init_pops #carrying capacity of each group
    
    n_pops = sim.N_Pop

    # All other params
    n_groups = length(K)
    ζ = Vector{Float32}(undef, n_groups) #latent rate
    γ = Vector{Float32}(undef, n_groups) #recovery/death rate
    μ_p = Vector{Float32}(undef, n_groups) #births/death rate at K
    
    ω = Vector{Float32}(undef, n_groups) #corpse infection modifier
    ρ = Vector{Float32}(undef, n_groups) #ASF mortality
    λ = Vector{Float32}(undef, n_groups) #corpse decay rate
    κ = Vector{Float32}(undef, n_groups) #waning immunity rate

    σ = Vector{Float32}(undef, n_pops) #density to non-density split for births and deaths
    θ = Vector{Float32}(undef, n_pops) #power of density effects for births/deaths 
    g = Vector{Float32}(undef, n_pops) #factor to allow for stable K with stochastic effects

    bw = Vector{Float32}(undef, n_pops)
    bo = Vector{Float32}(undef, n_pops)
    k = Vector{Float32}(undef, n_pops)
    la = Vector{Float32}(undef, n_pops)
    lo = Vector{Float32}(undef, n_pops)

    if sim.Model == 3
        n_pops  = counts.pop
    else
        n_pops = 1
    end
    
    for i in 1:n_pops
        
        data =  pops[i]
        data_s = sea[i]

        nf = counts.feral[i]
        nl = counts.farm[i]
        nt = counts.total[i]
        
        cs = counts.cum_sum
        
        σ[i] = data.Density_rate[1]
        θ[i] = data.Density_power[1]
        g[i] = data.g_fit[1]


        if sim.Identical#if running off means

            ζ[cs[i]+1:cs[i+1]] .= data.Latent[1]
            γ[cs[i]+1:cs[i+1]] .= data.Recovery[1]
            ρ[cs[i]+1:cs[i+1]] .= data.Death[1]
            κ[cs[i]+1:cs[i+1]] .= data.Immunity[1]
            λ[cs[i]+1:cs[i]+nf] .= data.Decay_f[1]
            λ[cs[i]+nf+1:cs[i+1]] .= data.Decay_l[1]
            
            birth_rate = 0.5*data.LN[1]*data.LS[1]*(1-data.LM[1])/365
            μ_p[cs[i]+1:cs[i+1]] .= birth_rate

        else #running of distros
            
            ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2], 0, 5) #latent dist
            γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2], 0, 5) #r/d rate dist
            ρ_d = TruncatedNormal(data.Death[1], data.Death[2], 0, 1) #mortality dist
            λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2], 0, 1) #corpse decay feral dist
            λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2], 0, 5) #corpse decay farm dist
            κ_d = TruncatedNormal(data.Immunity[1], data.Immunity[2], 0, 1)
            
            LN_d = TruncatedNormal(data.LN[1], data.LN[2], 0, 5) #number of yearly litters
            LS_d = TruncatedNormal(data.LS[1], data.LS[2], 1, 20) #litter size
            LM_d = TruncatedNormal(data.LM[1], data.LM[2], 1, 20) #litter mortality rate
            
            ζ[cs[i]+1:cs[i+1]] = rand(ζ_d,nt)
            γ[cs[i]+1:cs[i+1]] = rand(γ_d,nt)
            ω[cs[i]+1:cs[i+1]] = rand(ω_d,nt)
            ρ[cs[i]+1:cs[i+1]] = rand(ρ_d,nt)
            κ[cs[i]+1:cs[i+1]] = rand(κ_d,nt)
            λ[cs[i]+1:cs[i]+nf] .= rand(λ_fd,nf)
            λ[cs[i]+nf+1:cs[i+1]] .= rand(λ_ld,nl)

            
            μ_p[cs[i]+1:cs[i+1]] =  0.5*rand(LN_d,nt) .* rand(LS_d,nt) .* (1 .- rand(LM_d,nt)) ./ 365
            
        end


        if !sim.Fitted
            if sim.Identical
                ω[cs[i]+1:cs[i+1]] .= data.Corpse[1]
            else
                ω_d = TruncatedNormal(data.Corpse[1], data.Corpse[2], 0, 1) #corpse inf dist
                ω[cs[i]+1:cs[i+1]] = rand(ω_d,nt)
            end
        else #running off fitted!
            if sim.Model == 3
                if sim.Network == "s"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Scale_Free/omega.csv"
                elseif sim.Network == "w"
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Small_Worlds/omega.csv"
                else 
                    path = "Inputs/Fitted_Params/Tau-Hetrogeneous/Random/omega.csv"
                end
            elseif sim.Model == 2
                path = "Inputs/Fitted_Params/Tau-Homogeneous/omega.csv"
            else #ODE
                path = "Inputs/Fitted_Params/ODE/omega.csv"
            end 

            rand_values = rand(1:10000,1)
            df_omega = Array(CSV.read(path, DataFrame, header=false))
            omega  = df_omega[rand_values[1]]
            ω[cs[i]+1:cs[i+1]] .= omega
        end

        if sim.Seasonal

            bw[i] = data_s.Birth_width
            bo[i] = data_s.Birth_offset
            la[i] = data_s.Decay_amp
            lo[i] = data_s.Decay_offset

            k[i]  = birthpulse_norm(data_s.Birth_width, mean(μ_p[cs[i]+1:cs[i+1]]))

        else

            bw[i] = 0
            bo[i] = 0
            k[i]  = 0
            la[i] = 0
            lo[i] = 0

        end
        
    end

    return  μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, g, bw, bo, k, la, lo
    
end



function read_inputs(path, verbose)
    #=
    Function to read in the data for the tau simulation. Expecting a file for simulation meta data, 
    a folder with population data and another folder with seasonal data
    
    Inputs:
    -path, the path to directory with data for suns

    Outputs:
    -simulation, the simulation meta data
    -pops, data on n populations used in the model
    -seasonal, seaonal data on the n populations 
    =#
    Simulation = CSV.read("$(path)/Simulation_Data.csv", DataFrame; comment="#") #reading in simulation meta data
    
    Sim = Meta_Data(Simulation, verbose)
    N_P_files = size(readdir("$(Path)/Population"))[1] #number of different population region files we are reading in
        
    #number of population input files!
    Pops = Vector{Population_Data}(undef, N_P_files) #number of different regions we are population regions
    Seasons = Vector{Seasonal_effect}(undef, N_P_files) #number of different seasons
    
    for i in 1:N_P_files
        pop_data = CSV.read("$(path)/Population/Population_$(i).csv", DataFrame; comment="#") 
        Pops[i] = Population_Data(pop_data, Sim, verbose)

        if Sim.Seasonal
            seasonal_data = CSV.read("$(path)/Seasonal/Seasonal_$(i).csv", DataFrame; comment="#")
            Seasons[i] = Seasonal_effect(seasonal_data, Pops[i], verbose)
        end

    end
    
    return Sim, Pops, Seasons
    
end 


function birthpulse_norm(s, DT)
   
    integral, err = quadgk(x -> exp(-s*cos(pi*x/year)^2), 1, year, rtol=1e-8);
    k = (year*DT)/integral 
    
    return k
end

end
