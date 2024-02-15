using JLD2
using DelimitedFiles

base = "/home/callum/Desktop/ASF_Output/ACT/4000/Line_16/"

function endemiccalc(tp)

    data = load_object(tp);
    n_sims = length(data);

    np = size(data[1][1])[1]

    out = zeros(Int16, n_sims)

    if np == 1
        for i in 1:n_sims
            sim_data = data[i][1][1]
            slice = sim_data[1:1826]
            slice[slice .< 0] .= 0
            if slice[end] > 0
                out[i] = -1
            else
                d_day = findfirst( slice .==0 )
                out[i] = d_day
            end
        end
    else
        for i in 1:n_sims
            sim_data = data[i][1]

            total = sum(hcat(sim_data...), dims =2)

            total[total .< 0] .= 0
            if total[end] > 0
                out[i] = -1
            else
                d_day = findfirst( total .==0 )
                out[i] = d_day[1]
            end
        end
    end

    return out[out.>0], sum(out .== -1)
end


function main(sm)
	if sm ==1 
		Reg  = endemiccalc(base*"Regular/out.jdl2")
		writedlm( base*"Regular/die.csv",  Reg[1], ',')
	elseif sm ==2 
		Low  = endemiccalc(base*"Low/out.jdl2")
		writedlm( base*"Low/die.csv",  Low[1], ',')
	else
		High = endemiccalc(base*"High/out.jdl2")
		writedlm( base*"High/die.csv",  High[1], ',')
	end
end

