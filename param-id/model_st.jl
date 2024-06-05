using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames, RigidBodyDynamics, CSV, DataFrames

include("../../utils/torque_references/references.jl")
include("../../utils/recorded_data/load_real_data.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time

@register_symbolic (f::UFunction)(t)

u_functions = []
for i in eachindex(dfs)
    push!(u_functions, UFunction(dfs[i], 1)) # Select between 3 methods
end

# External torque function (as before, consider any changes if necessary)
function torque_ext(t, kt; exp = 1)
    # voltage = waveform_value_at_time(t, ref)
    u_function = u_functions[exp]
    voltage = u_function(t)
    # voltage = current_for_time(t)
    return voltage * kt
end

function modeler(mechanism;experiment = 1)
    @variables t
    sts = @variables τ(t) = 0. θ(t) = 0. ω(t) = 0.
    D = Differential(t)
    par = @parameters τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305 str_scale = 0. q̇_s = 0.1

    # str_scale = (τ_s - τ_c)
    β = exp(-abs(ω/q̇_s))
    w_coul = q̇_s / 10
    τ_f = (τ_c + str_scale * β) * tanh(ω / w_coul) + Kv * ω

    eqs = [
        τ ~ torque_ext(t, kt; exp = experiment)
        D(θ) ~ ω
        D(ω) ~ (τ - τ_f - dynamics_bias(MechanismState(mechanism, [θ], [ω]))[1]) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
        ] # (τ_s >= τ_c) is there to avoid that the optimization algorithm find a higher value for tau_c than for tau_s (which is not physically correct)
    @named sys = ODESystem(eqs, t, sts, par)
    sys = structural_simplify(sys)
    return sys
end

function simulation(sys; u0 = [0., 0.0], tspan = (0.0, 60.0), pars = [0.128, 1.2465, 0.2843, 0.07305, 0.128, 0.1])
    prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_c => pars[1], sys.Kv => pars[2], sys.kt => pars[3], sys.J_motor => pars[4], sys.str_scale => pars[5], sys.q̇_s => pars[6]])
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol
end

function plot_simulation(sol; experiment = 1)
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity [rad/s]")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position [rad]")
    plot(p1, p2, layout=(2,1), link=:x, leg = false, xlabel = "time [s]")
end

function plot_simu_vs_real(sol; experiment = 1)
    df = dfs[experiment]
    p2 = plot(sol.t, [u[1] for u in sol.u], ylabel="    Position [rad]", label="Simulation", color="blue", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p2, df.timestamp, df[!, "θ(t)"], color="red", label="Real data", linewidth=1)

    p0 = plot(sol.t, [torque_ext(t, 0.2843; exp = experiment) for t in sol.t], ylabel="    Torque [Nm]", label="Torque", color="green", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p0, df.timestamp, df.U * 2.6657 / 9.3756, color="red", label="Real data", linewidth=2)

    plot(p0, p2, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
end