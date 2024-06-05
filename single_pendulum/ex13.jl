include("model_urdf.jl")
include("double_pendulum/double_pendulum_system.jl")

mechanism = get_mechanism(;wanted_mech = "double_pendulum")

# Estimate the next state from a given state u0 = [q_1, q̇_1, q_2, q̇_2], for a given input, in a given time tspan
input = [0., .4]
tspan = 1.
stateVectorInit = [0., 0., 0., 0.] # [q_1, q̇_1, q_2, q̇_2]
sys, model = double_pendulum_system.system(mechanism; constant_input = input)
stateVectorNext = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan) # then get the next state
prob = double_pendulum_system.problem(sys, constant_input=input)

using BenchmarkTools
# Once the system is defined, it is no more necessary to do so
input = [1., .2]
tspan = .02
stateVectorInit = [0., 0., 0., 0.]
# @btime stateVectorNext0 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# Speed-Up
sstateVectorNext0 = double_pendulum_system.get_next_state_speed_up(sys, prob, stateVectorInit, input, tspan)

# sstateVectorNext0 = double_pendulum_system.get_next_state_speed_up2(sys, prob, stateVectorInit, input, tspan)


# # Once the system is defined, it is no more necessary to do so
# input = [1., .2]
# tspan = .1
# stateVectorInit = [0., 0., 0., 0.]
# @btime stateVectorNext1 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# # Speed-Up
# @btime sstateVectorNext1 = double_pendulum_system.get_next_state_speed_up(sys, prob, stateVectorInit, input, tspan)

# # Once the system is defined, it is no more necessary to do so
# input = [1., .2]
# tspan = 1.
# stateVectorInit = [0., 0., 0., 0.]
# @btime stateVectorNext2 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# # Speed-Up
# @btime sstateVectorNext2 = double_pendulum_system.get_next_state_speed_up(sys, prob, stateVectorInit, input, tspan)

# # Once the system is defined, it is no more necessary to do so
# input = [1., .2]
# tspan = 10.
# stateVectorInit = [0., 0., 0., 0.]
# @btime stateVectorNext3 = double_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)

# # Speed-Up
# @btime sstateVectorNext3 = double_pendulum_system.get_next_state_speed_up(sys, prob, stateVectorInit, input, tspan)


# stateVectorNext0, sstateVectorNext0
# stateVectorNext1, sstateVectorNext1
# stateVectorNext2, sstateVectorNext2
# stateVectorNext3, sstateVectorNext3

# # We can also linearise the system around a given operation point
# operation_point = [0., 0., 0., 0.]
# matrices = double_pendulum_system.linear_system(sys, model, operation_point)

# matrices hold (; A, B, C, D)