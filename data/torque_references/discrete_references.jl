module discrete_references
using CSV
using DataFrames

df1 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df2 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df3 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df4 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df5 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df6 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)

function get_u_at_time(t_target; ref = 0)
    if ref == 0
        df = df1
    end
    
    # Find the largest time point in 't' that is less than or equal to 't_target'
    idx = findlast(df.t .<= t_target)
    
    # If 't_target' is before the first time point, return 0.
    if idx === nothing
        return 0.
    end
    
    # Return the corresponding 'U' value
    return df.U[idx]
end
end