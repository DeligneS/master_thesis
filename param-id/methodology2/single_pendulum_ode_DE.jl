module single_pendulum_ode_DE

using DifferentialEquations, Plots, Parameters, StaticArrays, RigidBodyDynamics

############### The module simulate a single-pendulum urdf with control in voltage ###############
# ...

# Include using a relative path
include("../../utils/torque_references/references.jl")
include("../../utils/torque_references/discrete_references.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .discrete_references: get_u_at_time

@enum PendulumState STATIC MOVING_CW MOVING_CCW
global state
global shoulder
global torque_ref
global state_history = []

@with_kw mutable struct PendulumParams
    tau_s::Float64 = 0.128 # Static friction torque in Nm
    tau_c::Float64 = 0.128 # Coulomb friction torque in Nm
    Kv::Float64 = .22 # Viscous damping coefficient in Nm*s
    ke::Float64 = 3.6103 # back emf constant
    Ra::Float64 = 9.3756 # Armature resistance
    kt::Float64 = 2.6657 # Torque constant
    fsm_state::PendulumState = STATIC # Initial state of the fsm
end

# External torque function
function torque_ext(t)
    kt = 2.6657 #* 1.15
    Ra = 9.3756
    voltage = waveform_value_at_time(t, torque_ref)
    voltage = get_u_at_time(t)
    return voltage * kt / Ra  # Convert voltage to torque
    # return 0 # No external torque applied
end

# Differential equation model for pendulum dynamics
function pendulum_ode!(du, u, p, t)
    theta, omega = u
    @unpack_PendulumParams p

    q = u[1] # u[1:num_positions(state)]
    ω = u[2] # u[(num_positions(state) + 1):end]

    # Update derivatives based on state
    if fsm_state == STATIC #&& abs(tau_g + torque_ext(t)) <= tau_s
        du[1] = 0.
        du[2] = 0.
    elseif fsm_state == MOVING_CW
        set_configuration!(state, shoulder, q)
        set_velocity!(state, shoulder, ω)
        result = DynamicsResult(state.mechanism)
        tau_net = torque_ext(t) - Kv * ω - kt * ke * ω / Ra + tau_c
        dynamics!(du, result, state, u, [tau_net])
    else # fsm_state == MOVING_CCW
        set_configuration!(state, shoulder, q)
        set_velocity!(state, shoulder, ω)
        result = DynamicsResult(state.mechanism)
        tau_net = torque_ext(t) - Kv * ω - kt * ke * ω / Ra - tau_c
        dynamics!(du, result, state, u, [tau_net])
    end
end


function pendulum_conditions(out, u, t, integ)
    @unpack_PendulumParams integ.p
    theta, omega = u

    q = u[1] # u[1:num_positions(state)]
    ω = u[2] # u[(num_positions(state) + 1):end]
    set_configuration!(state, shoulder, q)
    set_velocity!(state, shoulder, ω)
    v̇ = copy(velocity(state))
    v̇[1] = 0
    τ = -inverse_dynamics(state, v̇)[1]
    
    # Conditions for fsm_state transitions
    out[1] = (fsm_state == STATIC) * ((τ + torque_ext(t)) - tau_s)
    out[2] = (fsm_state == STATIC) * (-(τ + torque_ext(t)) - tau_s)
    out[3] = (fsm_state == MOVING_CW) * (ω)
    out[4] = (fsm_state == MOVING_CCW) * (-ω)
    out[5] = (fsm_state == STATIC) && (τ > tau_s)
    out[6] = (fsm_state == STATIC) && (τ < -tau_s)
end


function pendulum_affects!(integ, idx)
    @unpack_PendulumParams integ.p
    theta, omega = integ.u
    t = integ.t
    q = integ.u[1] # u[1:num_positions(state)]
    ω = integ.u[2] # u[(num_positions(state) + 1):end]
    # set_configuration!(state, shoulder, q)
    # set_velocity!(state, shoulder, ω)
    v̇ = copy(velocity(state))
    v̇[1] = 0
    τ = -inverse_dynamics(state, v̇)[1]

    if (idx == 1) || (idx == 5)
        integ.p.fsm_state = MOVING_CCW
    elseif idx == 2 || (idx == 6)
        integ.p.fsm_state = MOVING_CW
    elseif idx == 3
        if (τ + torque_ext(t)) > tau_s
            integ.p.fsm_state = MOVING_CCW
        else
            integ.p.fsm_state = STATIC
            integ.u[2] = 0.0
        end
    elseif idx == 4
        if (τ + torque_ext(t)) < -tau_s
            integ.p.fsm_state = MOVING_CW
        else
            integ.p.fsm_state = STATIC
            integ.u[2] = 0.0
        end
    end
    if integ.p.fsm_state == STATIC
        push!(state_history, (t, 0))
    elseif integ.p.fsm_state == MOVING_CCW
        push!(state_history, (t, 1))
    elseif integ.p.fsm_state == MOVING_CW
        push!(state_history, (t, -1))
    end
end

function build_problem(state_, shoulder_;u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref_="triangle")
    # Note : avec l'urdf : 0 est stable, et avec la dynamique calculée à la main c'est pi qui est stable
    cb = VectorContinuousCallback(pendulum_conditions, pendulum_affects!, nothing, 6)

    p = PendulumParams()
    @unpack_PendulumParams p

    global state = state_
    global shoulder = shoulder_
    global torque_ref = torque_ref_

    set_configuration!(state, shoulder, u0[1])
    set_velocity!(state, shoulder, u0[2])
    v̇ = copy(velocity(state))
    v̇[1] = 0
    τ = -inverse_dynamics(state, v̇)[1]
    if τ > tau_s
        p.fsm_state = MOVING_CCW
    elseif τ < - tau_s
        p.fsm_state = MOVING_CW
    else 
        p.fsm_state = STATIC
    end
    # print(p.fsm_state)
    prob = ODEProblem(pendulum_ode!, u0, tspan, p)
    sol = solve(prob, callback = cb, dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol, state_history
end

function plot_simulation(sol)
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    p0 = plot(sol.t, [torque_ext(t) for t in sol.t], ylabel = "torque, Nm", label="Torque")
    plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
end

end