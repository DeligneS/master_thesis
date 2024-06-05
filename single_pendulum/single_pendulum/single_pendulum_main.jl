using CSV, DataFrames
include(joinpath("..", "model_urdf.jl"))
include("model_callback.jl")
include("model_tanh.jl")

mechanism = get_mechanism(; wanted_mech = "single_pendulum")

# Validation set : 9 -> 14
experiment = 10

sol, prob, sys = model_tanh.build_problem(mechanism; experiment = experiment, change_tspan = true, tspan = (0.0, 120.0))
model_tanh.plot_simulation(sol; experiment = experiment)
