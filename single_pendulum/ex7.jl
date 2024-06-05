using ModelingToolkit, Plots, DataFrames, RigidBodyDynamics, DifferentialEquations
using ModelingToolkit: t_nounits as t, D_nounits as D

include("model_urdf.jl")
include(joinpath("double_pendulum", "double_pendulum_system.jl"))

mechanism = get_mechanism(;wanted_mech = "double_pendulum")

# Estimate the next state from a given state u0 = [q_1, q̇_1, q_2, q̇_2], for a given input, in a given time tspan
input = [0., .4]
tspan = 5.
stateVectorInit = [0., 0., 0., 0.] # [q_1, q̇_1, q_2, q̇_2]
sys, model = double_pendulum_system.system(mechanism; constant_input = input)
stateVectorNext = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan) # then get the next state

# Once the system is defined, it is no more necessary to do so
input = [1., .2]
tspan = 1.
stateVectorInit = [0., 0., 0., 0.]
using BenchmarkTools
@btime stateVectorNext2 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# Once the system is defined, it is no more necessary to do so
input = [1., .2]
tspan = .1
stateVectorInit = [0., 0., 0., 0.]
@btime stateVectorNext2 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# Once the system is defined, it is no more necessary to do so
input = [1., .2]
tspan = 5.
stateVectorInit = [0., 0., 0., 0.]
@btime stateVectorNext2 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# We can also linearise the system around a given operation point
operation_point = [0., 0., 0., 0.]
matrices = double_pendulum_system.linear_system(sys, model, operation_point)

# matrices hold (; A, B, C, D)