include("model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
using CSV, DataFrames
mechanism = get_mechanism(;wanted_mech = "single_pendulum")

u0 = [0., 0.0]
tspan = (.0, .2)
input = .0001
sys, model = single_pendulum_system.system(mechanism, constant_input=input)
prob = single_pendulum_system.problem(sys, constant_input=input, u0 = u0, tspan = tspan)
# sstateVectorNext0 = single_pendulum_system.get_next_state_speed_up(sys, prob, u0, input, tspan[2])
# print(sstateVectorNext0)

# Solve the ODE
sol = solve(prob, Tsit5(), reltol=1e-3, abstol=1e-3)

print(sol[end])
# Plotting the results
p0 = plot(sol, vars=(0,1), label=["x₁" "x₂"], title="State Evolution")
# plot!(sol.t, input_func.(sol.t), label="Input", linetype=:steppost) # Plot input 

# Create a DataFrame with your data
df = DataFrame(time = sol.t, pos_sim = [u[1] for u in sol.u], vel_sim = [u[2] for u in sol.u])

# Write the DataFrame to a CSV file
# CSV.write("../utils/recorded_data/simulation/nonsmooth/tanh1.csv", df)

plot(p0, link=:x, size=(800, 600), xlabel="Time [s]")