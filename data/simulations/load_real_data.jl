using CSV
using DataFrames, Interpolations

dfs = []
# Firt calibration set (no more used)
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_chirp_05_200.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_chirp_075_2.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_chirp_increasing_075_2.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_chirp_increasing_075_3.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_sin_in_sin_1.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_sin_in_sin_075.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_sin_in_sin_in_sin.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp_processed/M1_sin_in_sin.csv", DataFrame))

# Validation set
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_square.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_tooth.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_triangle.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_square_1_5.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_tooth_1_5.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_triangle_1_5.csv", DataFrame))

# Calibration set
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/calibration_processed/chirp_1.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/calibration_processed/non_trivial_2.csv", DataFrame))

# Walking ref set
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/xing_trajectories/pwm_ctrl_fast.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/xing_trajectories/pwm_ctrl_slow.csv", DataFrame))

# Current ref set
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/current_exp/exp1_square_12.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/current_exp/exp1_tooth_12.csv", DataFrame))
push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/current_exp/exp1_triangle_12.csv", DataFrame))


### Definition of the functions
struct UFunction
    df::DataFrame
    method::Union{Int, Nothing}
end

function (f::UFunction)(t)
    # If no method is predefined, choose a default method, or use a decision logic
    method_to_use = isnothing(f.method) ? 1 : f.method

    if method_to_use == 1
        return method1(f, t)
    elseif method_to_use == 2
        return method2(f, t)
    elseif method_to_use == 3
        return method3(f, t)
    elseif method_to_use == 4
        return method4(f, t)
    elseif method_to_use == 5
        return method1_on_current(f, t)
    else
        error("Invalid method selected")
    end
end

# Method1: Returns the last known value of U at or before time t.
function method1(f::UFunction, t)
    idx = findlast(f.df.timestamp .<= t)
    if idx === nothing
        return 0.
    end
    return f.df.U[idx]
end

# Method2: Linearly interpolates the value of U at time t if t is between two timestamps.
function method2(f::UFunction, t)
    idx_before = findlast(f.df.timestamp .<= t)
    if idx_before === nothing
        return 0.
    end

    if f.df.timestamp[idx_before] == t
        return f.df.U[idx_before]
    end

    idx_after = findfirst(f.df.timestamp .> t)
    if idx_after === nothing
        return f.df.U[idx_before]
    end

    time_before = f.df.timestamp[idx_before]
    time_after = f.df.timestamp[idx_after]
    U_before = f.df.U[idx_before]
    U_after = f.df.U[idx_after]
    U_interpolated = U_before + (U_after - U_before) * ((t - time_before) / (time_after - time_before))
    return U_interpolated
end

# Method3: Similar to Method2 but handles edge cases for indices and returns interpolated U.
function method3(f::UFunction, t)
    idx = findlast(f.df.timestamp .<= t)
    if idx === nothing
        return 0.
    elseif idx == length(f.df.timestamp) || f.df.timestamp[idx] == t
        return f.df.U[idx]
    else
        t0 = f.df.timestamp[idx]
        t1 = f.df.timestamp[idx + 1]
        U0 = f.df.U[idx]
        U1 = f.df.U[idx + 1]
        return U0 + ((U1 - U0) / (t1 - t0)) * (t - t0)
    end
end

# Method4: Uses spline interpolation to describe well the transient between each measured value of U.
function method4(f::UFunction, t)
    # Ensure that the timestamps are sorted and the corresponding U values are properly aligned
    sorted_indices = sortperm(f.df.timestamp)
    times = f.df.timestamp[sorted_indices]
    values = f.df.U[sorted_indices]

    # Check if the requested time is outside the range of the timestamps
    if t < first(times) || t > last(times)
        return "Time out of bounds"
    end

    # Create a spline interpolation object
    itp = interpolate((times,), values, Gridded(Linear()))

    # Interpolate to find the value at the requested time t
    return itp(t)
end


# Method1: Returns the last known value of the current at or before time t.
function method1_on_current(f::UFunction, t)
    idx = findlast(f.df.timestamp .<= t)
    if idx === nothing
        return 0.
    end
    return f.df.DXL_Current[idx]
end