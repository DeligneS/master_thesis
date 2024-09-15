# Example of use

# We first include the urdf and the system definition
include("../URDF/model_urdf.jl")
include("single_pendulum/single_pendulum_system.jl")

# Define the single-pendulum mechanism (RigidBodyDynamics)
mechanism = get_mechanism(;wanted_mech = "single_pendulum")

# Estimate the next state from a given state u0 = [q, q̇], for a given input, in a given time tspan
input = 2.
tspan = 5.
stateVectorInit = [0., 0.]
sys, model = single_pendulum_system.system(mechanism, constant_input=input) # First define the system
# stateVectorNext = single_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan) # then get the next state

# Once the system is defined, it is no more necessary to do so
input = 1.
tspan = 1.
stateVectorInit = [0., 0.]
using BenchmarkTools
@btime stateVectorNext2 = single_pendulum_system.get_next_state(sys, stateVectorInit, input, tspan)
print(stateVectorNext2)

prob = single_pendulum_system.problem(sys, constant_input=input)
@btime stateVectorNext2 = single_pendulum_system.get_next_state_speed_up(sys, prob, stateVectorInit, input, tspan)
print(stateVectorNext2)

input = 1.
tspan = 1.
uk = [0., 0.]
prob = ODEProblem(sys, [sys.J_total.q => uk[1], sys.J_total.q̇ => uk[2]], (0., tspan), [sys.v_input.U => input])
sol = solve(prob, Rosenbrock23(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

# using SciMLStructures: Tunable, replace, replace!

newprob = remake(prob, tspan = (0., tspan); u0 = [sys.J_total.q => uk[1], sys.J_total.q̇ => uk[2]], p = [sys.v_input.U => input])
sol = solve(newprob, Rosenbrock23(), dtmax=0.01, reltol=1e-3, abstol=1e-3)

print("Ok")