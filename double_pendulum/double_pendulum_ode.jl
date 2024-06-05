module double_pendulum_ode

using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse
using ModelingToolkit: t_nounits as t, D_nounits as D
using RigidBodyDynamics

# Include using a relative path
include("../../modelisation/torque_references/references.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time
global state
global shoulder
global mechanism

# External torque function (as before, consider any changes if necessary)
function torque_ext(t; torque_ref = "triangle")
    kt = 2.6657 * 1.15
    Ra = 9.3756
    voltage = waveform_value_at_time(t, torque_ref)
    # voltage = current_for_time(t)
    return voltage * kt / Ra
    # return 0.1
end

function compute_dynamics(mechanism, θ, ω)
    actual_state = MechanismState(mechanism, [θ], [ω])
    v̇ = copy(velocity(state))
    v̇[1] = 0
    τ_dynamics = -inverse_dynamics(actual_state, v̇)[1]
    return τ_dynamics
end

function affect!(integ, u, p, ctx)
    if (ctx == 0) # STATIC
        integ.ps[p.fsm_state] = 0
        integ.u[u.ω] = 0.
    elseif (ctx == 1) # MOVING
        integ.ps[p.fsm_state] = 1
    end
end

@register_symbolic torque_ext(t)

@mtkmodel SP begin
    @parameters begin
        τ_s = 0.128 
        τ_c = 0.128 
        Kv = 0.22 
        ke = 3.6103 
        Ra = 9.3756 
        kt = 2.6657
        J_motor = 0.07305
        fsm_state = 0
        ωmin = 10e-6
    end
    @variables begin
        τ(t) = torque_ext(0.)
        θ(t) = 0.
        ω(t) = 0.0
    end
    @structural_parameters begin
        h = 1
    end
    @equations begin
        τ ~ torque_ext(t)
        D(θ) ~ ω
        D(ω) ~ fsm_state * (compute_dynamics(mechanism, θ, ω) + τ - sign(ω) * τ_c - (kt * ke / Ra + Kv) * ω) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
    end
    @discrete_events begin
        ((fsm_state == 1) * (abs(ω) < ωmin)) => (affect!, [ω], [fsm_state], [], 0)
        ((fsm_state == 0) * (abs(compute_dynamics(mechanism, θ, ω) + torque_ext(t)) > τ_s)) => (affect!, [], [fsm_state], [], 1)
    end
end

function build_problem(mechanism_, state_, shoulder_)
    global state = state_
    global shoulder = shoulder_
    global mechanism = mechanism_
    @mtkbuild sp = SP()
    matrices, simplified_sys = linearize(sp, [sp.τ], [sp.θ])
    prob = ODEProblem(sp, [], (0.0, 120.0), [])
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol
end

function plot_simulation(sol)
    # println(sol.ps[-1])
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    # torques = [torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    p0 = plot(sol.t, torques, ylabel = "torque, Nm", label="Torque")
    plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
end

end