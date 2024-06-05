include("model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
include("../utils/recorded_data/load_real_data.jl")
using DataFrames, CSV, Dates  # Ensure all needed modules are used

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

# Estimate the next state from a given state and input
input = 0.
tspan = 0.02  # The time step increment
prediction_horizon = 5  # Number of steps to predict ahead
sys, model = single_pendulum_system.system(mechanism; constant_input = input)
prob = single_pendulum_system.problem(sys, constant_input=input)

# Start timing the loop
# start_time = now()

experiments = [9, 10, 11, 12, 13, 14, 17, 18]
list1 = 1:19
list2 = 20:5:45
# list3 = 50:10:200
prediction_horizons = vcat(list1, list2)

for prediction_horizon in prediction_horizons
    # Start timing the loop
    start_time = now()
    for experiment in experiments
        data = dfs[experiment]
        predicted_data = DataFrame(timestamp = Float64[], predicted_position = Float64[], predicted_velocity = Float64[])

        idxs = axes(data, 1)  # Get the valid indices of data
        i = first(idxs)  # Start at the first index

        while i <= (last(idxs) - prediction_horizon)
            current_state = [data[i, :"θ(t)"], data[i, :DXL_Velocity]]
            # Predict 'prediction_horizon' steps ahead
            for step in 1:prediction_horizon
                next_index = i + step - 1
                if next_index > last(idxs)
                    break  # Break if we exceed the number of available data points
                end
                # Calculate the next state
                current_input = data[next_index, :U]
                next_state = single_pendulum_system.get_next_state_speed_up(sys, prob, current_state, current_input, tspan)
                current_state = next_state  # Update current state for the next step
                if step == prediction_horizon
                    # Append the predicted state to the dataframe
                    push!(predicted_data, (timestamp = data[i+prediction_horizon, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
                end
            end
            # Increment index by prediction horizon for the next reset point
            i += 1
        end
        # Save the predictions to a new CSV file
        CSV.write("../utils/recorded_data/validation/method2/hrz$(prediction_horizon)_exp$(experiment).csv", predicted_data)
    end
    # Calculate and print the elapsed time
    elapsed_time = now() - start_time
    println("Time taken for the loop: $(elapsed_time)")
end

