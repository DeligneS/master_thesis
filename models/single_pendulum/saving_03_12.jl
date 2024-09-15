module single_pendulum_ode

using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse
using ModelingToolkit: t_nounits as t, D_nounits as D
using RigidBodyDynamics

# Include using a relative path
include("../../data/torque_references/references.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
global state
global shoulder
global mechanism
# @enum PendulumState STATIC MOVING_CW MOVING_CCW

# @with_kw mutable struct PendulumParams
#     tau_s::Float64 = 0.128 # Static friction torque in Nm
#     fsm_state::PendulumState = STATIC # Initial state of the fsm
# end

# External torque function (as before, consider any changes if necessary)
function torque_ext(t; torque_ref = "triangle")
    kt = 2.6657 * 1.15
    Ra = 9.3756
    voltage = waveform_value_at_time(t, torque_ref)
    return voltage * kt / Ra
    # return 0.
end

function compute_dynamics(mechanism, θ, ω)
    actual_state = MechanismState(mechanism, [θ], [ω])
    # set_configuration!(state, shoulder, θ)
    # set_velocity!(state, shoulder, ω)
    v̇ = copy(velocity(state))
    v̇[1] = 0
    τ_dynamics = -inverse_dynamics(actual_state, v̇)[1]
    return τ_dynamics
end

function compute_net_torque(t, ω, Kv, kt, ke, Ra, tau_c)
    return - Kv * ω - kt * ke * ω / Ra - sign(ω) * tau_c
end


# @mtkmodel FOL begin
#     @parameters begin
#         tau_s = 0.128 
#         tau_c = 0.128 
#         Kv = 0.22 
#         ke = 3.6103 
#         Ra = 9.3756 
#         kt = 2.6657
#         ωmin = 0.001
#     end
#     @variables begin
#         θ(t) = pi/2
#         ω(t) = 0.0
#     end
#     @equations begin
#         D(θ) ~ ω
#         D(ω) ~ (compute_net_torque(t, ω, Kv, kt, ke, Ra, tau_c) + compute_dynamics(mechanism, θ, ω)) / (mass_matrix(state)[1])
#         # D(ω) ~ IfElse.ifelse(signbit.(compute_net_torque(t, ω, Kv, kt, ke, Ra, tau_c) + compute_dynamics(state) - tau_s), 0, (compute_net_torque(t, ω, Kv, kt, ke, Ra, tau_c) + compute_dynamics(state)) / (mass_matrix(state)[1]))
#         # D(ω) ~ compute_torque(θ, ω, state, tau_c, tau_s, Kv, ωmin, t, kt, ke, Ra) / (mass_matrix(state)[1])
#     end
# end

# function affect!(integ, u, p, ctx)
#     if ctx == 0
#         integ.p[p] = -1
#     elseif ctx == 1
#         integ.p[p] = 1
#     elseif ctx == 2 # STATIC
#         integ.p[p] = 0
#     end
# end

# function affect!(integ, u, p, ctx)
#     if (ctx == 0) #& (integ.u[u.fsm_state] == 0)
#         integ.ps[p.fsm_state] = 1
#     elseif (ctx == 1) #& (integ.u[u.fsm_state] == 0) # MOVING_CW
#         integ.ps[p.fsm_state] = -1
#     elseif (ctx == 2) & (integ.ps[p.fsm_state] == 1) # STATIC
#         integ.ps[p.fsm_state] = 0
#     elseif (ctx == 3) & (integ.ps[p.fsm_state] == -1) # STATIC
#         integ.ps[p.fsm_state] = 0
#     end
# end

# function affect!(integ, u, p, ctx)
#     if (ctx == 0) & (integ.u[u.fsm_state] == 0)
#         integ.u[u.fsm_state] = 1
#     elseif (ctx == 1) & (integ.u[u.fsm_state] == 0) # MOVING_CW
#         integ.u[u.fsm_state] = -1
#     elseif (ctx == 2) & (integ.u[u.fsm_state] != 0) # STATIC
#         integ.u[u.fsm_state] = 0
#     end
# end

function affect!(integ, u, p, ctx)
    if (ctx == 0) #& (integ.u[u.fsm_state] == 0)
        integ.u[u.fsm_state] = 1
    elseif (ctx == 1) #& (integ.u[u.fsm_state] == 0) # MOVING_CW
        integ.u[u.fsm_state] = -1
    elseif (ctx == 2) #& (integ.u[u.fsm_state] == 1) # STATIC
        integ.u[u.fsm_state] = 0
    elseif (ctx == 3) & (integ.u[u.fsm_state] == -1) # STATIC
        integ.u[u.fsm_state] = 0
    end
end

@mtkmodel FOL begin
    @parameters begin
        τ_s = 0.128 
        τ_c = 0.128 
        Kv = 0.22 
        ke = 3.6103 
        Ra = 9.3756 
        kt = 2.6657
        ωmin = 0.001
        # fsm_state = 1
    end
    @variables begin
        θ(t) = 0.0
        ω(t) = 0.0
        fsm_state(t) = 0
    end
    @equations begin
        # D(θ) ~ IfElse.ifelse(fsm_state == 0, 0., ω)
        # D(ω) ~ IfElse.ifelse(fsm_state == 0, 0., # STATIC state, no movement
        #                     IfElse.ifelse(fsm_state == 1, # MOVING_CCW
        #                                 (compute_dynamics(mechanism, θ, ω) + torque_ext(t) - kt * ke * ω / Ra - Kv*ω - τ_c)/(mass_matrix(MechanismState(mechanism, [θ], [ω]))[1]),
        #                                 (compute_dynamics(mechanism, θ, ω) + torque_ext(t) - kt * ke * ω / Ra - Kv*ω + τ_c)/(mass_matrix(MechanismState(mechanism, [θ], [ω]))[1]))) # MOVING_CW
        # D(fsm_state) ~ 0.
        D(θ) ~ ω
        D(ω) ~ (compute_dynamics(mechanism, θ, ω) + torque_ext(t) - kt * ke * ω / Ra - Kv*ω - sign(ω) * τ_c) / (mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
        
        # D(θ) ~ IfElse.ifelse(fsm_state == 0, 0., ω)
        # D(ω) ~ IfElse.ifelse(fsm_state == 0, 0., # STATIC state, no movement
        #                     compute_dynamics(mechanism, θ, ω) + torque_ext(t) - kt * ke * ω / Ra - Kv*ω - sign(ω) * τ_c)/(mass_matrix(MechanismState(mechanism, [θ], [ω]))[1]) # MOVING_CCW
        D(fsm_state) ~ 0.
    end
    @continuous_events begin
        # [(1 - abs(fsm_state)) * (torque_ext(t)) ~ τ_s] => (affect!, [], [fsm_state], [], 0)   # Transition to MOVING_CCW
        # [(1 - abs(fsm_state)) * (-(torque_ext(t))) ~ τ_s] => (affect!, [], [fsm_state], [], 1)  # Transition to MOVING_CW
        # [(ω ~ 0)] => (affect!, [], [fsm_state], [], 2)      # Transition to STATIC
        # [(-ω ~ 0)] => (affect!, [], [fsm_state], [], 3)     # Transition to STATIC
        # [(1 - abs(fsm_state)) * (compute_dynamics(mechanism, θ, ω) + torque_ext(t)) ~ τ_s] => (affect!, [fsm_state], [], [], 0)   # Transition to MOVING_CCW
        # [(1 - abs(fsm_state)) * (-(compute_dynamics(mechanism, θ, ω) + torque_ext(t))) ~ τ_s] => (affect!, [fsm_state], [], [], 1)  # Transition to MOVING_CW
        # [(ω ~ 0)] => (affect!, [fsm_state], [], [], 2)      # Transition to STATIC
        # [(-ω ~ 0)] => (affect!, [fsm_state], [], [], 3)     # Transition to STATIC
        [ω ~ 0]
    end
end

function build_problem(mechanism_, state_, shoulder_)
    global state = state_
    global shoulder = shoulder_
    global mechanism = mechanism_
    # p = PendulumParams()
    @mtkbuild fol = FOL()
    prob = ODEProblem(fol, [], (0.0, 120.0), [])
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol
end

function plot_simulation(sol)
    p1 = plot(sol.t, [sol.u[i][3] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "position, rad")
    plot!(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    # mylist = [compute_dynamics(mechanism, el[1], el[2]) for el in sol.u]
    torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    # torques = [torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    p0 = plot(sol.t, torques, ylabel = "torque, Nm", label="Torque")
    plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
end

end