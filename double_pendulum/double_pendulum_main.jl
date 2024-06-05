include("double_pendulum_urdf.jl")
include("double_pendulum_ode.jl")

mechanism, state, shoulder = dp_urdf.double_pendulum_mechanism()
sol = double_pendulum_ode.build_problem(mechanism, state, shoulder)
double_pendulum_ode.plot_simulation(sol)
