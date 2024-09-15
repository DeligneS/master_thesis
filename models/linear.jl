include("model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
using CSV, DataFrames, ModelingToolkit, ModelingToolkitStandardLibrary, Dates
mechanism = get_mechanism(;wanted_mech = "single_pendulum")

u0 = [0., 0.0]
tspan = (.0, .02)
input = 10.2
sys, model = single_pendulum_system.system(mechanism, constant_input=input)
lin_fun, simplified_sys = linearization_function(model, :i, :o) # simplify = false, initialize = true, kwargs...)

sys, model = single_pendulum_system.system(mechanism, constant_input=input, stribeck=true)
prob = single_pendulum_system.problem(sys, constant_input=input, u0 = u0, tspan = tspan)

using DifferentialEquations, Plots
# Define the system dynamics function
function linear_system!(dx, x, p, t)
    u = p(t)  # Get the input value at time t
    dx .= A * x + B * u
end

inputs = vcat(-12:0.5:-0.5, 0.5:0.5:12)
tspans = 0.02:0.02:0.2
states_pos = -1.5:0.1:1.5
states_vel = -2:0.2:2

data = DataFrame(
    init_state = Vector{Float64}[],  
    input = Float64[],               
    tspan = Float64[],               
    lin_state = Vector{Float64}[],    
    nl_state = Tuple{Float64, Float64}[]
)
for pos in states_pos
    for vel in states_vel
        # Start timing the loop
        start_time = now()
        op = Dict(sys.J_total.q => pos, sys.J_total.q̇ => vel)
        (; A, B, C, D) = linearize(simplified_sys, lin_fun; op = op)
        for input in inputs
            for tspan in tspans
                x0 = [pos, vel]
                input_func(t) = input
                prob2 = ODEProblem(linear_system!, x0, (0.0, tspan), input_func)
                sol = solve(prob2, Tsit5(), reltol=1e-5, abstol=1e-5)
                lin_state = sol[end]
                nl_state = single_pendulum_system.get_next_state_speed_up(sys, prob, x0, input, tspan)
                push!(data, (init_state = x0, input = input, tspan = tspan, lin_state = lin_state, nl_state = nl_state))
                # print("-")
            end
        end
        # Calculate and print the elapsed time
        elapsed_time = now() - start_time
        println("Time taken for the loop: $(elapsed_time)")
    end
end
CSV.write("linear.csv", data)

# using LinearAlgebra

# function next_state_zoh(A, B, x, u, h) # Suitable for constant input
#     return exp(A * h) * x + inv(A) * (exp(A * h) - I) * B * u
# end

# x = u0

# h = 0.1

# x_next_zoh = next_state_zoh(A, B, x, [input], h)

# print(x_next_zoh)

# Initial state and time span
x0 = u0  # Initial state

# Define the input function (example: step input)
input_func(t) = input

# Create the ODE problem
prob = ODEProblem(linear_system!, x0, tspan, input_func)

# Solve the ODE
sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-5)

print(sol[end])
# # # Plotting the results
# # p0 = plot(sol, vars=(0,1), label=["x₁" "x₂"], title="State Evolution")
# # # plot!(sol.t, input_func.(sol.t), label="Input", linetype=:steppost) # Plot input 

# # # Create a DataFrame with your data
# # df = DataFrame(time = sol.t, pos_sim = [u[1] for u in sol.u], vel_sim = [u[2] for u in sol.u])

# # # Write the DataFrame to a CSV file
# # # CSV.write("../utils/recorded_data/simulation/nonsmooth/tanh1.csv", df)

# # plot(p0, link=:x, size=(800, 600), xlabel="Time [s]")