module model_tanh

# using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames
using ModelingToolkit, Plots, DataFrames, RigidBodyDynamics, DifferentialEquations
using ModelingToolkit: t_nounits as t, D_nounits as D

# Include using a relative path
include("../../utils/torque_references/references.jl")
include("../../utils/recorded_data/load_real_data.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time
global mechanism
global state_history::Vector{Tuple{Float64, Float64}} = []

@register_symbolic (f::UFunction)(t)

u_functions = []
for i in eachindex(dfs)
    push!(u_functions, UFunction(dfs[i], 2)) # Select between 3 methods
end

# External torque function (as before, consider any changes if necessary)
function torque_ext(t, kt; exp = 1)
    u_function = u_functions[exp]
    voltage = u_function(t)
    return voltage * kt
end

function compute_dynamics(mechanism, θ, ω)
    actual_state = MechanismState(mechanism, [θ], [ω])
    v̇ = copy(velocity(actual_state))
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

@mtkmodel SP begin
    @structural_parameters begin
        experiment = 1
        stribeck = false
    end
    @parameters begin
        τ_c = 0.128 
        Kv = 1.2465
        kt = 0.2843
        J_motor = 0.07305
        τ_brk = 0.128 
        w_brk = 0.01
    end
    @variables begin
        τ(t) = 0.
        θ(t), [output = true]
        ω(t), [output = true]
    end
    begin
        if stribeck
            str_scale = sqrt(2 * exp(1)) * (τ_brk - τ_c)
            w_st = w_brk * sqrt(2)
            w_coul = w_brk / 10
            τ_f = str_scale * (exp(-(ω / w_st)^2) * ω / w_st) + τ_c * tanh(ω / w_coul) + Kv * ω
        else
            # τ_f = τ_c * tanh(ω / 1e-4) + Kv * ω
            τ_f = τ_c * sign(ω) + Kv * ω
        end
    end
    @equations begin
        τ ~ torque_ext(t, kt; exp = experiment)
        D(θ) ~ ω
        D(ω) ~ (compute_dynamics(mechanism, θ, ω) + τ - τ_f) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
    end
end

function build_problem(mechanism_; u0 = [0., 0.0], change_tspan = false, tspan = (0.0, 120.0), pars = [0.0706396353290064, 1.5651123281135342, 0.3952914552970312, 0.040786221279949124, 0.19756590009983593, 0.009441932814455819], experiment=1, stribeck=false)
    global mechanism = mechanism_
    if !change_tspan
        tspan = (dfs[experiment].timestamp[1], dfs[experiment].timestamp[end])
    end
    @mtkbuild sp = SP(experiment = experiment, stribeck = stribeck)
    # matrices, simplified_sys = linearize(sp, [sp.τ], [sp.θ])
    prob = ODEProblem(sp, [sp.θ => u0[1], sp.ω => u0[2]], tspan, [sp.τ_c => pars[1], sp.Kv => pars[2], sp.kt => pars[3], sp.J_motor => pars[4], sp.τ_brk => pars[5], sp.w_brk => pars[6]])
    sol = solve(prob, Tsit5(), dtmin=0.005)#, reltol=1e-5, abstol=1e-5)
    return sol, prob, sp
end

function plot_simulation(sol; experiment = 1)
    # println(sol.ps[-1])
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity [rad/s]")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position [rad]")
    # torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    torques = [torque_ext(t, 0.2843; exp = experiment) for t in sol.t]
    # p0 = plot(sol.t, torques, ylabel = "torque, Nm", label="Torque")
    # plot(p0, p1, p2, layout=(3,1), link=:x, leg = false)
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

end