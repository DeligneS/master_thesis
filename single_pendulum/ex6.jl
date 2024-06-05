using ModelingToolkit, Plots, DataFrames, RigidBodyDynamics, DifferentialEquations
using ModelingToolkit: t_nounits as t, D_nounits as D

include("model_urdf.jl")
include(joinpath("double_pendulum", "double_pendulum_system.jl"))

mechanism = get_mechanism(;wanted_mech = "double_pendulum")

experiment = [17, 17]
pars = [0.06510461345450586, 1.5879662676966781, 0.39454422423683916, 0., 0.06510461345450586, 0.]
stribeck = false
experiment = [0., .4]
constant_input = true
sys, model = double_pendulum_system.system(mechanism; pars, stribeck, experiment = experiment, constant_input = constant_input)

u0 = [0., 0., 0., 0.]
tspan = (0., 5)
prob = double_pendulum_system.problem(sys; tspan = tspan, experiment = experiment, constant_input = constant_input)
sol = solve(prob, Rosenbrock23(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

# p0 = plot(sol.t, sol[sys.J_total.q̈], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
p0 = plot(sol.t, sol[sys.J_total.q], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
p1 = plot(sol.t, sol[sys.J_total2.q], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

plot(p0,p1, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")

# sol.u
# (sol.u[end][1], sol.u[end][2], sol.u[end][3], sol.u[end][4])