include("model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
include("../utils/recorded_data/load_real_data.jl")
using DataFrames, CSV, Dates, Plots

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

# Estimate the next state from a given state and input
input = 0.0
tspan = 0.02 

experiment = 9

data = dfs[experiment]

# --- Prediction ---
pars = [0.128, 1.247, 0.284, 0., 0., 0.07305]
pars = [0.10767283437834642, 1.3543917039372881, 0.38855092682229664, 0.038593539900614364, 0.0008419984873591663, 0.005714079304930255]
pars = [0.06446650303355177, 1.5892897697716262, 0.3951923010983331, 0.00909605814894139, 0., 0.]

sys, model = single_pendulum_system.system(mechanism; pars=pars, constant_input = input, stribeck=false)
prob = single_pendulum_system.problem(sys, constant_input=input)


idxs = axes(data, 1)  
i = 1  
prediction_horizon = size(data, 1)
global ntm
# Initialize current_state OUTSIDE the loop to avoid the scope issue
# current_state = [data[i, :"θ(t)"], data[i, :DXL_Velocity]]
x = 0
experiments = [9, 10, 11, 12, 13, 14, 17, 18]
for experiment in experiments
    data = dfs[experiment]
    prediction_horizon = size(data, 1)
    predicted_data = DataFrame(timestamp = Float64[], predicted_position = Float64[], predicted_velocity = Float64[])
    ntm = [data[1, :"θ(t)"], data[1, :DXL_Velocity]]
    for step in 1:prediction_horizon
        current_input = data[step, :U]
        next_state = single_pendulum_system.get_next_state_speed_up(sys, prob, ntm, current_input, tspan)
        # Explicitly state it's a local variable
        ntm = next_state
    
        push!(predicted_data, (timestamp = data[step, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
    end
    # Save the predictions to a new CSV file
    CSV.write("../utils/recorded_data/validation/method1/model2'/hrzinf_exp$(experiment).csv", predicted_data)
end

# --- Simulation ---
sp, model = single_pendulum_system.system(mechanism; experiment = experiment)
u0 = [0., 0.]
prob = single_pendulum_system.problem(sp; experiment = experiment)
sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

# --- Plotting ---
p0 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel="Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
plot!(p0, predicted_data.timestamp, predicted_data.predicted_position, ylabel="Position [rad]", label="Position", color="red", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
