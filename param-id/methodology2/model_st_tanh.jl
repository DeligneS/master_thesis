using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, DataFrames, RigidBodyDynamics, CSV, DataFrames
# using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames, RigidBodyDynamics, CSV, DataFrames

include("../../data/load_real_data.jl")

# Using the module or its functions
# using .torque_references: waveform_value_at_time
# using .torque_references: current_for_time

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

function modeler(mechanism; u0 = [0., 0.0], tspan = (0.0, 60.0), experiment = 1)
    @variables t
    sts = @variables τ(t) = 0. θ(t) = 0. ω(t) = 0.
    D = Differential(t)
    par = @parameters τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305 fsm_state = 1 sign_velocity = 1
    eqs = [
        τ ~ torque_ext(t, kt; exp = experiment)
        D(θ) ~ ω
        D(ω) ~ fsm_state * (τ - (τ_c * tanh(ω / 1e-4) + Kv * ω) - dynamics_bias(MechanismState(mechanism, [θ], [ω]))[1]) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
        ]
    @named sys = ODESystem(eqs, t, sts, par)
    sys = structural_simplify(sys)
    # prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_c => tau_c, sys.Kv => K_v, sys.kt => k_t])
    # sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sys
end

function simulation(sys; u0 = [0., 0.0], tspan = (0.0, 60.0), pars = [0.128, 1.2465, 0.2843, 0.07305])
    prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_c => pars[1], sys.Kv => pars[2], sys.kt => pars[3], sys.J_motor => pars[4]])
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