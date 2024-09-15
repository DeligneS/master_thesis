using Plots, DataFrames, RigidBodyDynamics, DifferentialEquations, CSV

include("../URDF/model_urdf.jl")
include(joinpath("single_pendulum", "single_pendulum_system.jl"))

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

experiment = 13 #17 - 18 for walking gait
pars = [0.06510461345450586, 1.5879662676966781, 0.39454422423683916, 0., 0.06510461345450586, 0.]
stribeck = false
sp, model = single_pendulum_system.system(mechanism; pars, stribeck, experiment = experiment)

u0 = [0., 0.]
prob = single_pendulum_system.problem(sp; experiment = experiment)

using BenchmarkTools
# @btime sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3, save_everystep = false)
# sol = solve(prob, Rosenbrock23(), dtmax=0.01, reltol=1e-3, abstol=1e-3)
sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

# operation_point = [0., 0.]
# matrices = single_pendulum_system.linear_system(sp, model, operation_point)

# input = 2.
# tspan = 5.
# sys, model = single_pendulum_system.system(mechanism; pars, stribeck, constant_input = true)
# next_u = single_pendulum_system.get_next_state(sys, u0, input, tspan)


# p0 = plot(sol.t, sol[sp.J_total.q̈], ylabel="    Acceleration [rad/s^2]", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
p0 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.v_input.output.u], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

# p0 = plot(sol.t, sol[sp.ctrl.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.τ_grav.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.τ_f.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.J_total.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")

# Create a DataFrame with your data
# df = DataFrame(time = sol.t, acc_sim = sol[sp.J_total.q̈], pos_sim = [u[1] for u in sol.u], vel_sim = [u[2] for u in sol.u], input_V = sol[sp.v_input.output.u], input_tau = sol[sp.ctrl.joint_out.τ], tau_grav = sol[sp.τ_grav.joint_out.τ], tau_f = sol[sp.τ_f.joint_out.τ])

# Write the DataFrame to a CSV file
# CSV.write("../utils/recorded_data/simulation/validation/exp3_tooth_1_5.csv", df)

# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_square.csv", DataFrame))
# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_tooth.csv", DataFrame))
# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp2_triangle.csv", DataFrame))
# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_square_1_5.csv", DataFrame))
# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_tooth_1_5.csv", DataFrame))
# push!(dfs, CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/validation/PWM_Processed/exp3_triangle_1_5.csv", DataFrame))
