include("model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")
include("double_pendulum/double_pendulum_system.jl")

include("../utils/recorded_data/load_real_data.jl")
using RigidBodyDynamics, DifferentialEquations, Plots

experiment = 4
mechanism = get_mechanism(;wanted_mech = "single_pendulum")
mechanism = get_mechanism(;wanted_mech = "double_pendulum")

# sp, model = single_pendulum_system.system(mechanism; experiment = experiment)
# u0 = [0., 0.]
# prob = single_pendulum_system.problem(sp; experiment = experiment)
# sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

# p0 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # p0 = plot(sol.t, sol[sp.v_input.output.u], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

# # p0 = plot(sol.t, sol[sp.ctrl.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # p0 = plot(sol.t, sol[sp.τ_grav.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # p0 = plot(sol.t, sol[sp.τ_f.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # p0 = plot(sol.t, sol[sp.J_total.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # plot!(p0, dfs[experiment].timestamp, dfs[experiment][!, "θ(t)"], ylabel="    Torque [Nm]", label="Torque", color="red", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # plot!(p0, sol.t, sol[sp.τ_grav.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# plot!(p0, dfs[experiment].timestamp, dfs[experiment][!, "θ(t)"], ylabel="    Position [rad]", label="Position", color="red", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

# plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")

actual_state = MechanismState(mechanism)
# set_configuration!(actual_state, [0., pi/2])
# set_configuration!(actual_state, [pi/2, pi/2])
set_configuration!(actual_state, [pi/2])

zero_velocity!(actual_state)

v̇ = similar(velocity(actual_state))
fill!(v̇, 0.0)
τ_dynamics = -inverse_dynamics(actual_state, v̇)
C = dynamics_bias(actual_state)

# elseif hip
#     τ_grav = compute_dynamics(mechanism, [q, other_joint.q], [q, other_joint.q])
# else # knee of double_pendulum
#     τ_grav = compute_dynamics(mechanism, [other_joint.q, q], [other_joint.q, q])
# end


println("The gravitational torque is: ", C)
println("The torque computed by inverse dynamics is: ", τ_dynamics)
println("The mass matrix is: ", mass_matrix(actual_state))
