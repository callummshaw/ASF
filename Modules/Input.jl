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
    N_ensemble::Int16 #number of runs in an ensemble
    Identical::Bool #if we params to be drawn from dist or just means of dist
    Seasonal::Bool #If model is seasonal
    N_Pop::Int8 #number of populations, must match the number of population files in input
    N_Inf::Int8 #number of populations init with ASF
    N_Seed::Int8 #population we will seed with
    C_Type::String #what kind of connection we want to init with between populations, line (l), circular (c), total (t), or off (o)
    C_Str::Float32 #strength of the connections between populations, note this is only for l,c,t
    Network::String #Type of network used for model random (r), scale-free (s), or small worlds (w)
    N_param::Float32 #Network parameter (only for small worlds)
    
    function Meta_Data(input, numv, verbose)
       
        Mn = parse(Int8, input.Value[1])
       
        F = (input.Value[2] == "true")
       
        Ny = parse(Int16, input.Value[3])
      
        Ne = parse(Int16,input.Value[4])
       
        I  = (input.Value[5] == "true")
        S  = (input.Value[6] == "true")
        Np = parse(Int8, input.Value[7])
        Ni = parse(Int8, input.Value[8])
        Ns = parse(Int16, input.Value[9])
        Ct = input.Value[10]
        Cs = parse(Float32, input.Value[11])
        Nw = input.Value[12]
        Nps= parse(Float32, input.Value[13])
        Sd = parse(Int16, input.Value[14])
        
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
        end

        new(Mn,F,Ny,Sd,Ne,I,S,Np,Ni,Ns,Ct,Cs,Nw,Nps)
        
    end
end

struct Population_Data <: Data_Input
    #=
    Structure to store parameters for each population
    =#

    Dense::Vector{Float64} #density of population
    N_feral::Vector{UInt16} #number of feral groups
    N_farm::Vector{UInt8} #number of farm groups
    N_f::Vector{Float64} #feral group size
    N_l::Vector{Float64} #farm population size
    Boar_p::Vector{Float64} #Proportion of groups that are wild boars
    N_int::Vector{UInt8} #average interconnection between feral groups
    N_e::Vector{Float64} #% of population seeded in exposed
    N_i::Vector{Float64} #% of population seeded in infected
    Birth::Vector{Float64} #natural birth rate, at K births equal deaths
    Density_rate::Vector{Float32} #percent of natural birth/deaths that are NOT related to population density (0-1)
    Density_power::Vector{Float32} #power of K/N or N/K use for density birth/death rates
    g_fit::Vector{Float32} #fitting param to ensure stable birth/death rate
    Death::Vector{Float64} #ASF death prob
    B_f::Vector{Float64} #intra feral group transmission
    B_l::Vector{Float64} #intra farm transmission
    B_ff::Vector{Float64} #inter feral group transmission
    B_fl::Vector{Float64} #farm-feral transmission
    B_Density::Vector{Float64} #density dependence of beta (0 for frequncy, 1 for density)
    Corpse::Vector{Float64} #corpse infection modifier
    Latent::Vector{Float64} #latent period
    Recovery::Vector{Float64}#Recovery rate
    Immunity::Vector{Float64} #immunity
    Decay_f::Vector{Float64} #decay feral
    Decay_l::Vector{Float64} #decay farm
    
    function Population_Data(input, verbose)
        Den = [input.Mean[1],input.STD[1]] 
        Nf = [input.Mean[2],input.STD[2]]
        Nl = [input.Mean[3],input.STD[3]]
        Sf = [input.Mean[4],input.STD[4]]
        Sl = [input.Mean[5],input.STD[5]]
        Br = [input.Mean[6],input.STD[6]]
        Ni = [input.Mean[7],input.STD[7]]
        Npe = [input.Mean[8],input.STD[8]]
        Npi = [input.Mean[9],input.STD[9]]
        B = [input.Mean[10],input.STD[10]]
        Dr = [input.Mean[11],input.STD[11]]
        Dp = [input.Mean[12],input.STD[12]]
        Gf = [input.Mean[13],input.STD[13]]
        D = [input.Mean[14],input.STD[14]]
        Bf = [input.Mean[15],input.STD[15]]
        Bl = [input.Mean[16],input.STD[16]]
        Bff = [input.Mean[17],input.STD[17]]
        Bfl = [input.Mean[18],input.STD[18]]
        Bd = [input.Mean[19],input.STD[19]]
        C = [input.Mean[20],input.STD[20]]
        L = day_to_rate(input.Mean[21],input.STD[21])
        R = day_to_rate(input.Mean[22],input.STD[22])
        Im = day_to_rate(input.Mean[23],input.STD[23])
        Df = [input.Mean[24],input.STD[24]]
        Dl = [input.Mean[25],input.STD[25]]
        
        #Here just checking the inputs to make sure they are reasonable/expected
        if verbose 
            if Den[1] > 10
                @warn "High starting density of $(Den[1])"
            end

            if (Nf[1] > 1000) | (Nf[1] < 100)
                @warn "$(Nf[1]) feral groups"
            end 

            if (Sf[1] > 25) | (Sf[1]<4)
                @warn "Mean feral group size of $(Sf[1])"
            end
            
            @info "Boar to group ratio of $(Br[1])"
            @info "Mean feral connectivity of $(Ni[1])"
            

            if (B[1] > 0.01) | (B[1] < 0.001)
                @warn "Birth rate of $(B[1])"
            end  

            if (Dr[1] > 1 ) | (Dr[1] < 0)
                @warn "Density_rate must be between 0-1"
            end 

            if (Bd[1] > 1 ) | (Bd[1] < 0)
                @warn "β_density outside of 0-1"
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
     
        new(Den, Nf, Nl, Sf, Sl, Br, Ni, Npe, Npi, B, Dr, Dp, Gf, D, Bf, Bl, Bff, Bfl, Bd, C, L, R, Im, Df, Dl)

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
    η::Vector{Float32} #ensity power for transmission
    g::Vector{Float32} #fitting param for stable populations

    Seasonal::Bool #if we are running with seasonality

    bw::Vector{Float32}
    bo::Vector{UInt8} 
    k::Vector{Float32} 
    la::Vector{Float32}
    lo::Vector{UInt8}

    Populations::Network.Network_Data #breakdown of population
    
    function Model_Parameters(sim, pops, sea, U0, Populations, network)
        
        β, connected_pops, connected_births = Beta.construction(sim, pops, Populations, network)
        μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, bw, bo, k, la, lo  = parameter_build(sim, pops, sea, U0, Populations)
        
        new(β, connected_births, connected_pops, μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, sim.Seasonal, bw, bo, k, la, lo, Populations)
    end
    
end

struct Model_Data
    #=
    Structure to store key data on mode
    =#
    MN::Int8 #which model we are running with (1- ODE, 2-Tau Homogeneous, 3-Tau Heterogeneous)
    Time::Tuple{Float32, Float32} #Model run time
    NR::Int16 #number of runs in ensemble!
    U0::Vector{Int32} #Initial Population
    Parameters::Model_Parameters #Model parameters
    Populations_data::Vector{Population_Data} #distributions for params

    function Model_Data(Path; verbose = false, fitting_rewire = 0)
        #custom_network parameter used for fitting!
        sim, pops, sea = read_inputs(Path, verbose)

        Time = (sim.S_day,sim.years*365+sim.S_day)

        #now building feral pig network
        network, counts = Network.build(sim, pops, fitting_rewire, verbose) 
        
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
    η = Vector{Float32}(undef, n_pops) #power of density effects for transmission
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
        η[i] = data.B_Density[1]
        g[i] = data.g_fit[1]


        if sim.Identical#if running off means

            ζ[cs[i]+1:cs[i+1]] .= data.Latent[1]
            γ[cs[i]+1:cs[i+1]] .= data.Recovery[1]
            μ_p[cs[i]+1:cs[i+1]] .= data.Birth[1]
            ρ[cs[i]+1:cs[i+1]] .= data.Death[1]
            κ[cs[i]+1:cs[i+1]] .= data.Immunity[1]
            λ[cs[i]+1:cs[i]+nf] .= data.Decay_f[1]
            λ[cs[i]+nf+1:cs[i+1]] .= data.Decay_l[1]

        else #running of distros
            
            ζ_d = TruncatedNormal(data.Latent[1], data.Latent[2], 0, 5) #latent dist
            γ_d = TruncatedNormal(data.Recovery[1], data.Recovery[2], 0, 5) #r/d rate dist
            μ_p_d = TruncatedNormal(data.Birth[1], data.Birth[2], 0, 1) #birth dist
            ρ_d = TruncatedNormal(data.Death[1], data.Death[2], 0, 1) #mortality dist
            λ_fd = TruncatedNormal(data.Decay_f[1], data.Decay_f[2], 0, 1) #corpse decay feral dist
            λ_ld = TruncatedNormal(data.Decay_l[1], data.Decay_l[2], 0, 5) #corpse decay farm dist
            κ_d = TruncatedNormal(data.Immunity[1], data.Immunity[2], 0, 1)

            ζ[cs[i]+1:cs[i+1]] = rand(ζ_d,nt)
            γ[cs[i]+1:cs[i+1]] = rand(γ_d,nt)
            ω[cs[i]+1:cs[i+1]] = rand(ω_d,nt)
            ρ[cs[i]+1:cs[i+1]] = rand(ρ_d,nt)
            κ[cs[i]+1:cs[i+1]] = rand(κ_d,nt)
            μ_p[cs[i]+1:cs[i+1]] =  rand(μ_p_d,nt) 
            λ[cs[i]+1:cs[i]+nf] .= rand(λ_fd,nf)
            λ[cs[i]+nf+1:cs[i+1]] .= rand(λ_ld,nl)

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
                    path = "../Inputs/Fitted_Params/Posteriors/Tau-Hetrogeneous/Scale_Free/omega.csv"
                elseif sim.Network == "w"
                    path = "../Inputs/Fitted_Params/Posteriors/Tau-Hetrogeneous/Small_Worlds/omega.csv"
                else 
                    path = "../Inputs/Fitted_Params/Posteriors/Tau-Hetrogeneous/Random/omega.csv"
                end
            elseif sim.Model == 2
                path = "../Inputs/Fitted_Params/Posteriors/Tau-Homogeneous/omega.csv"
            else #ODE
                path = "../Inputs/Fitted_Params/Posteriors/ODE/omega.csv"
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

    return  μ_p, K, ζ, γ, ω, ρ, λ, κ, σ, θ, η, g, bw, bo, k, la, lo
    
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
    
    n_inf  = 1#infected_populations(Simulation) #what population is seeded with ASF (custom population!)

    Sim = Meta_Data(Simulation, n_inf, verbose)
    
    Pops = Vector{Population_Data}(undef, Sim.N_Pop)
    Seasons = Vector{Seasonal_effect}(undef, Sim.N_Pop)
    
    for i in 1:Sim.N_Pop
        pop_data = CSV.read("$(path)/Population/Population_$(i).csv", DataFrame; comment="#") 
        Pops[i] = Population_Data(pop_data, verbose)

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