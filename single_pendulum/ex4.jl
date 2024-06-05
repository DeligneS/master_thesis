include("single_pendulum/single_pendulum_urdf.jl")

experiment = 18

global mechanism = single_pendulum.single_pendulum_mechanism()

using ModelingToolkit, Plots, DataFrames, RigidBodyDynamics, DifferentialEquations
using ModelingToolkit: t_nounits as t, D_nounits as D

include("model_components.jl")

# pars = [0.128, 1.2465, 0.2843, 0.07305, 0.128, 0]
pars = [0.06510461345450586, 1.5879662676966781, 0.39454422423683916, 0., 0.06510461345450586, 0.]
stribeck = false

@named v_input = VariableInput(experiment = experiment)
@named ctrl = OLController(kt = pars[3])
@named J_total = TotalInertia(J_motor = pars[4])
@named τ_f = FrictionTorque(stribeck = stribeck, τ_c = pars[1], Kv = pars[2], τ_s = pars[5], q̇_s = pars[6])
@named τ_grav = GravitationalTorque()
@named position_sensor = PositionSensor()
@named output_reader = Feedback1D()

connect(v_input.output, :i, ctrl.input_voltage)


# connections = [ connect(v_input.output, :i, ctrl.input_torque)
#                 connect(ctrl.joint_out, τ_grav.joint_in)
#                 connect(τ_grav.joint_out, τ_f.joint_in)
#                 connect(τ_f.joint_out,  J_total.joint_in)
#                 connect(J_total.joint_out, position_sensor.joint_in)
#                 connect(position_sensor.q, :o, output_reader.input)
#                 ]

# @named model = ODESystem(connections, t,
#                     systems = [
#                         v_input,
#                         ctrl,
#                         J_total,
#                         τ_f,
#                         τ_grav,
#                         position_sensor,
#                         output_reader
#                     ])
# sp = structural_simplify(model)

# u0 = [0., 0.]
# tspan = (dfs[experiment].timestamp[1], dfs[experiment].timestamp[end])
# # prob = ODEProblem(sp, [sp.J_total.q => u0[1], sp.J_total.q̇ => u0[2]], tspan, [])
# prob = ODEProblem(sp, [], tspan, [])

# using BenchmarkTools
# # @btime sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-3, abstol=1e-3, save_everystep = false)
# # sol = solve(prob, Rosenbrock23(), dtmax=0.01, reltol=1e-3, abstol=1e-3)
# sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)

# # my_op = [0., 0.]
# # op = Dict(sp.J_total.q => my_op[1], sp.J_total.q̇ => my_op[2])
# # matrices, simplified_sys = linearize(model, :i, :o, op=op)
# # println(matrices)

# # include("single_pendulum/model_callback_2.jl")
# # model_callback.plot_simu_vs_real(sol; experiment = experiment)

# ### Save simulation
# # Create a DataFrame with your data
# df = DataFrame(time = sol.t, acc_sim = sol[sp.J_total.q̈], pos_sim = [u[1] for u in sol.u], vel_sim = [u[2] for u in sol.u])

# # Write the DataFrame to a CSV file
# CSV.write("../utils/recorded_data/xing_trajectories/simu_slow_smooth_tsit.csv", df)

# p0 = plot(sol.t, sol[sp.J_total.q̈], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
# # p0 = plot(sol.t, sol[sp.ctrl.joint_out.τ], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

# plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")

# # # Named State Space
# # using ControlSystemsMTK

# # lsys = named_ss(model, :i, :o)

# # w = exp10.(LinRange(-4, 1, 200))
# # ControlSystemsMTK.bodeplot(lsys, w)

# # # Compute sensitivity in analysis
# # using ControlSystemsBase
# # using ModelingToolkitStandardLibrary.Blocks
# # matrices_S, simplified_sys = Blocks.get_sensitivity(model, :o)
# # So = ss(matrices_S...) |> minreal # The output-sensitivity function as a StateSpace system
# # matrices_T, simplified_sys = Blocks.get_comp_sensitivity(model, :o)
# # To = ss(matrices_T...)# The output complementary sensitivity function as a StateSpace system

# # w = exp10.(LinRange(-4, 10, 200))
# # bodeplot([So, To], w, label = ["S" "T"], plot_title = "Sensitivity functions",
# #     plotphase = false)