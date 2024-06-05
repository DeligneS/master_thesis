module single_pendulum_ode

using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames
using RigidBodyDynamics

#### Code for MTK v8.73.2 ####

# Include using a relative path
include("../../utils/torque_references/references.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time
global state
global shoulder
global mechanism
global state_history::Vector{Tuple{Float64, Float64}} = []

# External torque function (as before, consider any changes if necessary)
function torque_ext(t, kt; ref = "triangle")
    voltage = waveform_value_at_time(t, ref)
    # voltage = current_for_time(t)
    return voltage * kt
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
    elseif (ctx == 1) # MOVING_CCW
        integ.ps[p.fsm_state] = 1
        integ.ps[p.sign_velocity] = 1
    elseif (ctx == 2) # MOVING_CW
        integ.ps[p.fsm_state] = 1
        integ.ps[p.sign_velocity] = -1
    end
    push!(state_history, (integ.t, integ.ps[p.fsm_state] * integ.ps[p.sign_velocity]))
end

# @register_symbolic torque_ext(t)
# @mtkmodel Controller begin
#     @parameters begin
#         ref = "triangle"
#     end
#     @variables begin
#         τ(t) = 0.
#     end
#     @equations begin
#         τ ~ waveform_value_at_time(t, ref) * 2.6657 / 9.3756
#     end
# end
@variables t
D = Differential(t)
@mtkmodel SP begin
    @structural_parameters begin
        τ_ref = "triangle"
    end
    @parameters begin
        τ_s = 0.128 
        τ_c = 0.128 
        Kv = 0.22 + (2.6657 * 3.6103 / 9.3756)
        kt = 2.6657 / 9.3756
        J_motor = 0.07305
        fsm_state = 0
        sign_velocity = 0
        # τ_s# = 0.128 
        # τ_c# = 0.128 
        # Kv# = 0.22 + (2.6657 * 3.6103 / 9.3756)
        # kt# = 2.6657 / 9.3756
        # J_motor# = 0.07305
    end
    @variables begin
        τ(t) = 0.
        θ(t) = 0.#, [output = true]
        ω(t) = 0.#, [output = true]
    end
    @equations begin
        τ ~ torque_ext(t, kt; ref = τ_ref)
        D(θ) ~ ω
        D(ω) ~ fsm_state * (compute_dynamics(mechanism, θ, ω) + τ - sign_velocity * τ_c - Kv * ω) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
    end
    # @continuous_events begin
    #     [ω ~ 0] => (affect!, [ω], [fsm_state, sign_velocity], [], 0)
    # end
    # @discrete_events begin
    #     ((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = τ_ref)) > τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 1)
    #     ((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = τ_ref)) < -τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 2)
    # end
end

function build_problem(mechanism_, state_, shoulder_; u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref_="triangle")
    global state = state_
    global shoulder = shoulder_
    global mechanism = mechanism_

    @mtkbuild sp = SP(τ_ref = torque_ref_)
    continuous_events = [ω ~ 0] => (affect!, [ω], [fsm_state, sign_velocity], [], 0)
    discrete_events = [((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = τ_ref)) > τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 1)
                    ((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = τ_ref)) < -τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 2)
    ]
    # matrices, simplified_sys = linearize(sp, [sp.τ], [sp.θ])
    prob = ODEProblem(sp, [sp.θ => u0[1], sp.ω => u0[2]], tspan, []; )
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol, prob, sp
end


function identify_params(mechanism_, state_, shoulder_, df; u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref_="triangle")
    global state = state_
    global shoulder = shoulder_
    global mechanism = mechanism_
    @mtkbuild sys = SP(τ_ref = torque_ref_)
    # sp = structural_simplify(sp)
    # matrices, simplified_sys = linearize(sp, [sp.τ], [sp.θ])
    # local_id_all = assess_identifiability(sp)
    prob = ODEProblem(sys, [sp.θ => u0[1], sp.ω => u0[2]], tspan, [])
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    data = DataFrame(sol)
    trial = Trial(data, sys, tspan)
    invprob = InverseProblem([trial], sys, 
    [

    ]
    )
    return sol#, local_id_all
end


function plot_simulation(sol; torque_ref = "triangle")
    # println(sol.ps[-1])
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    # torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    torques = [torque_ext(t, 2.6657 / 9.3756; ref = torque_ref) for t in sol.t]
    p0 = plot(sol.t, torques, ylabel = "torque, Nm", label="Torque")
    plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
end

function plot_simu_vs_real(sol, df; torque_ref = "triangle")
    p2 = plot(sol.t, [u[1] for u in sol.u], ylabel="    Position [rad]", label="Simulation", color="blue", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p2, df.t, df.DXL_Position, color="red", label="Real data", linewidth=2)

    p0 = plot(sol.t, [torque_ext(t, 2.6657 / 9.3756; ref = torque_ref) for t in sol.t], ylabel="    Torque [Nm]", label="Torque", color="green", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p0, df.t, df.U * 2.6657 / 9.3756, color="red", label="Real data", linewidth=2)

    plot(p0, p2, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
end

end