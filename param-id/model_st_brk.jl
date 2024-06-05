using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames, RigidBodyDynamics, CSV, DataFrames

include("../../utils/torque_references/references.jl")
include("../../utils/recorded_data/load_real_data.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time

struct UFunction
    df::DataFrame
end

function (f::UFunction)(t)
    # Find the index of the largest time point in 't' that is less than or equal to 't'
    idx = findlast(f.df.timestamp .<= t)
    
    # If 't' is before the first time point, handle as needed
    if idx === nothing
        return 0. # Or adjust as needed for your specific requirements
    end
    
    # Return the corresponding 'U' value
    return f.df.U[idx]
end

@register_symbolic (f::UFunction)(t)

u_functions = []
for i in eachindex(dfs)
    push!(u_functions, UFunction(dfs[i]))
end

# External torque function (as before, consider any changes if necessary)
function torque_ext(t, kt; exp = 1)
    # voltage = waveform_value_at_time(t, ref)
    u_function = u_functions[exp]
    voltage = u_function(t)
    # voltage = current_for_time(t)
    return voltage * kt
end

function modeler(mechanism, state, ;u0 = [0., 0.0], tspan = (0.0, 60.0), experiment = 1)
    function compute_dynamics(mechanism, θ, ω)
        actual_state = MechanismState(mechanism, [θ], [ω])
        v̇ = copy(velocity(state))
        v̇[1] = 0
        τ_dynamics = -inverse_dynamics(actual_state, v̇)[1]
        return τ_dynamics
    end
    @variables t
    sts = @variables τ(t) = 0. θ(t) = 0. ω(t) = 0.
    D = Differential(t)
    par = @parameters τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305 τ_brk = 0.128 w_brk = 0.1 fsm_state = 1 sign_velocity = 1
    str_scale = IfElse.ifelse(τ_brk >= τ_c, sqrt(2 * exp(1)) * (τ_brk - τ_c), 0.)
    w_st = w_brk * sqrt(2)
    w_coul = w_brk / 10
    eqs = [
        τ ~ torque_ext(t, kt; exp = experiment)
        D(θ) ~ ω
        D(ω) ~ fsm_state * (compute_dynamics(mechanism, θ, ω) + τ - (str_scale * (exp(-(ω / w_st)^2) * ω / w_st) + τ_c * tanh(ω / w_coul) + Kv * ω)) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
        ]
    @named sys = ODESystem(eqs, t, sts, par)
    sys = structural_simplify(sys)
    # prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_c => tau_c, sys.Kv => K_v, sys.kt => k_t])
    # sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sys
end

function simulation(sys; u0 = [0., 0.0], tspan = (0.0, 60.0), pars = [0.128, 1.2465, 0.2843, 0.07305, 0.128, 0.1])
    prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_c => pars[1], sys.Kv => pars[2], sys.kt => pars[3], sys.J_motor => pars[4], sys.τ_brk => pars[5], sys.w_brk => pars[6]])
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol
end

function plot_simulation(sol, df; experiment = 1)
    # println(sol.ps[-1])
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    plot!(p2, df.timestamp, df[!, "θ(t)"], color="red", label="Real data", linewidth=1)
    # torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    torques = [torque_ext(t, 2.6657 / 9.3756; exp = experiment) for t in sol.t]
    p0 = plot(sol.t, torques, ylabel = "torque, Nm", label="Torque")
    plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
end
