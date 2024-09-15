include("../URDF/model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
include("../data/load_real_data.jl")
using DataFrames, CSV, Dates  # Ensure all needed modules are used

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

# Estimate the next state from a given state and input
input = 0.
tspan = 0.02  # The time step increment
prediction_horizon = 5  # Number of steps to predict ahead
pars = [0.128, 1.247, 0.284, 0., 0., 0.07305]
pars = [0.10767283437834642, 1.3543917039372881, 0.38855092682229664, 0.038593539900614364, 0.0008419984873591663, 0.005714079304930255]
pars = [0.06446650303355177, 1.5892897697716262, 0.3951923010983331, 0.00909605814894139, 0., 0.]
pars = [0.09075609155581482, 1.4603883441868397, 0.39447576410405144, 0.10000000972603594, 0., 0.] # Model 2 but with opti on pos + vel

sys, model = single_pendulum_system.system(mechanism; pars=pars, constant_input = input)
prob = single_pendulum_system.problem(sys, constant_input=input)

# Start timing the loop
# start_time = now()

experiments = [9, 10, 11, 12, 13, 14, 17, 18]
list1 = 1:19
list2 = 20:5:45
list3 = 50:10:200
prediction_horizons = vcat(list1, list2, list3)

for prediction_horizon in prediction_horizons
    # Start timing the loop
    start_time = now()
    for experiment in experiments
        data = dfs[experiment]
        predicted_data = DataFrame(timestamp = Float64[], predicted_position = Float64[], predicted_velocity = Float64[])

        idxs = axes(data, 1)  # Get the valid indices of data
        i = first(idxs)  # Start at the first index

        while i <= last(idxs)
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

                # Append the predicted state to the dataframe
                push!(predicted_data, (timestamp = data[next_index, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
            end
            # Increment index by prediction horizon for the next reset point
            i += prediction_horizon
        end
        # Save the predictions to a new CSV file
        CSV.write("../utils/recorded_data/validation/method1/model4/hrz$(prediction_horizon)_exp$(experiment).csv", predicted_data)
    end
    # Calculate and print the elapsed time
    elapsed_time = now() - start_time
    println("Time taken for the loop: $(elapsed_time)")
end

# for experiment in experiments
#     data = dfs[experiment]
#     predicted_data = DataFrame(timestamp = Float64[], predicted_position = Float64[], predicted_velocity = Float64[])

#     idxs = axes(data, 1)  # Get the valid indices of data
#     i = first(idxs)  # Start at the first index

#     while i <= last(idxs)
#         current_state = [data[i, :"θ(t)"], data[i, :DXL_Velocity]]
#         # Predict 'prediction_horizon' steps ahead
#         for step in 1:prediction_horizon
#             next_index = i + step - 1
#             if next_index > last(idxs)
#                 break  # Break if we exceed the number of available data points
#             end
#             # Calculate the next state
#             current_input = data[next_index, :U]
#             next_state = single_pendulum_system.get_next_state_speed_up(sys, prob, current_state, current_input, tspan)
#             current_state = next_state  # Update current state for the next step

#             # Append the predicted state to the dataframe
#             push!(predicted_data, (timestamp = data[next_index, :timestamp], predicted_position = next_state[1], predicted_velocity = next_state[2]))
#         end
#         # Increment index by prediction horizon for the next reset point
#         i += prediction_horizon
#     end
#     # Save the predictions to a new CSV file
#     CSV.write("../utils/recorded_data/validation/prediction_hrz$prediction_horizon/$experiment.csv", predicted_data)
# end


# # Calculate and print the elapsed time
# elapsed_time = now() - start_time
# println("Time taken for the loop: $(elapsed_time)")