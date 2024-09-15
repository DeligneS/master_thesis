include("../URDF/model_urdf.jl")
include("../data/load_real_data.jl")
include(joinpath("single_pendulum", "single_pendulum_system.jl"))
using DifferentialEquations, Plots, DataFrames

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

experiment = 20 #17 - 18 for walking gait
pars = [0.06510461345450586, 1.5879662676966781, 0.39454422423683916, 0., 0.06510461345450586, 0.] # For PWM control
pars = [0.06510461345450586, 1.5879662676966781 - (0.39454422423683916 * 3.61), 0.39454422423683916 * 9.3, 0., 0.06510461345450586, 0.]
# pars = [0., 0., 2.6657, 0., 0., 0.] # For current control without friction
# pars = [0.128, 0.22, 2.6657*2, 0., 0., 0.07305] # For current control with friction
pars = [0.128, 1.247, 0.284, 0., 0., 0.07305] # For PWM control with friction
# pars = [0.128, 1.247, 0.284*1.15, 0., 0., 0.07305] # For PWM control with friction + manual-tuned kt

stribeck = false
sp, model = single_pendulum_system.system(mechanism; pars, stribeck, experiment = experiment)

u0 = [0., 0.]
prob = single_pendulum_system.problem(sp; experiment = experiment)
sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

p0 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel="    Position [rad]", label="Position", color="blue", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
plot!(p0, dfs[experiment].timestamp, dfs[experiment].DXL_Position, ylabel="    Position [rad]", label="Position", color="red", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# plot!(p0, dfs[experiment].timestamp, dfs[experiment][!, "θ(t)"], ylabel="    Position [rad]", label="Position", color="red", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

# p0 = plot(sol.t, sol[sp.J_total.q̈], ylabel="    Acceleration [rad/s^2]", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel="    Velocity [Nm]", label="Velocity", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.v_input.output.u], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# plot!(p0, sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel="    Position [rad]", label="Position", color="blue", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.ctrl.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.τ_grav.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.τ_f.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# p0 = plot(sol.t, sol[sp.J_total.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# plot(p0, xlims=(0, 1), ylims=(-1,3), layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")


# dfs[experiment].
# Create a DataFrame with your data
df = DataFrame(time = sol.t, acc_sim = sol[sp.J_total.q̈], pos_sim = [u[1] for u in sol.u], vel_sim = [u[2] for u in sol.u], input_V = sol[sp.v_input.output.u], input_tau = sol[sp.ctrl.joint_out.τ], tau_grav = sol[sp.τ_grav.joint_out.τ] )#, tau_f = sol[sp.τ_f.joint_out.τ])

# Write the DataFrame to a CSV file
# CSV.write("../utils/recorded_data/simulation/current_analysis/pwm_ctrl+friction/triangle.csv", df)

plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")