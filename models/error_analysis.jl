include("../URDF/model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
include("../data/load_real_data.jl")
using Dates  # To use the DateTime function for better timing granularity

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

# Estimate the next state from a given state u0 = [q_1, q̇_1, q_2, q̇_2], for a given input, in a given time tspan
input = 0.
tspan = 1.
stateVectorInit = [0., 0.] # [q_1, q̇_1, q_2, q̇_2]
sys, model = single_pendulum_system.system(mechanism; constant_input = input)
prob = single_pendulum_system.problem(sys, constant_input=input)

# Start timing the loop
start_time = now()

experiments = [9, 10, 11, 12, 13, 14, 17, 18]
for experiment in experiments
    data = dfs[experiment]
    predicted_data = DataFrame(timestamp = Float64[], predicted_position = Float64[], predicted_velocity = Float64[])

    # Iterate through each row in the dataframe
    for i in eachindex(data[!, :timestamp])
        # Extract current state and input
        uk = [data[i, :"θ(t)"], data[i, :DXL_Velocity]]  # current state [position, velocity]
        input = data[i, :U]  # constant input
        tspan = 0.02

        # Get the predicted next state
        next_state = single_pendulum_system.get_next_state_speed_up(sys, prob, uk, input, tspan)
        # if i < size(data, 1)
        #     next_state = [data[i+1, :"θ(t)"], data[i+1, :DXL_Velocity]]
        #     # Append the predicted state to the dataframe
        #     push!(predicted_data, (timestamp = data[i, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
        # end

        # Append the predicted state to the dataframe
        push!(predicted_data, (timestamp = data[i, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
    end
    # Save the predictions to a new CSV file
    CSV.write("../utils/recorded_data/validation/prediction_hrz1/$experiment.csv", predicted_data)
end

# Calculate and print the elapsed time
elapsed_time = now() - start_time
println("Time taken for the loop: $(elapsed_time)")
