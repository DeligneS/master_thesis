module model_callback

### Please note : the sign(velocity) function is a tanh, because continuous/discrete events are not ? working with components (I don't achieve to make it work)

# using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames
using ModelingToolkit, Plots, DataFrames, RigidBodyDynamics, DifferentialEquations
using ModelingToolkit: t_nounits as t, D_nounits as D

# Include using a relative path
include("../../data/torque_references/references.jl")
include("../../data/load_real_data.jl")

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
    # return 0.
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
        integ.u[u.q̇] = 0.
    elseif (ctx == 1) # MOVING_CCW
        integ.ps[p.fsm_state] = 1
        integ.ps[p.sign_velocity] = 1
    elseif (ctx == 2) # MOVING_CW
        integ.ps[p.fsm_state] = 1
        integ.ps[p.sign_velocity] = -1
    end
    # push!(state_history, (integ.t, integ.ps[p.fsm_state] * integ.ps[p.sign_velocity]))
end

@connector Joint begin
    q(t), [description = "Rotation angle of joint"]
    τ(t), [connect = Flow, description = "Cut torque in joint"]
end

@mtkmodel OLController begin
    @structural_parameters begin
        experiment = 1
    end
    @components begin
        # joint_in = Joint()
        joint_out = Joint()
    end
    @parameters begin
        kt = 0.2843
    end
    @variables begin
        q(t), [description = "Absolute rotation angle", guess = 0.0]
        q̇(t), [description = "Absolute angular velocity", guess = 0.0]
        # q̈(t), [description = "Absolute angular acceleration", guess = 0.0]
    end
    @equations begin
        # q ~ joint_in.q
        q ~ joint_out.q
        D(q) ~ q̇
        joint_out.τ ~ torque_ext(t, kt; exp = experiment)
        # D(q̇) ~ q̈
    end
end

@mtkmodel Inertia begin
    @parameters begin
        J_motor = 0.07305, [description = "Motor's Moment of inertia"]
    end
    @components begin
        joint_in = Joint()
        # joint_out = Joint()
    end
    # begin
    #     @symcheck J_motor > 0 || throw(ArgumentError("Expected `J` to be positive"))
    # end
    @variables begin
        q(t), [description = "Absolute rotation angle", guess = 0.0]
        q̇(t), [description = "Absolute angular velocity", guess = 0.0]
        q̈(t), [description = "Absolute angular acceleration", guess = 0.0]
    end
    # begin
    #     J_total = J_motor + mass_matrix(MechanismState(mechanism, [q], [q̇]))[1]
    # end
    @equations begin
        q ~ joint_in.q
        # q ~ joint_out.q
        D(q) ~ q̇
        D(q̇) ~ q̈
        q̈ ~ joint_in.τ / (J_motor + mass_matrix(MechanismState(mechanism, [q], [q̇]))[1]) #+ joint_out.τ
    end
end

@mtkmodel FrictionTorque begin
    @structural_parameters begin
        stribeck = false
    end
    @components begin
        joint_in = Joint()
        joint_out = Joint()
    end
    @parameters begin
        τ_c = 0.128, [description = "Coulomb torque"]
        Kv = 1.2465, [description = "Viscous coefficient"]
        τ_s = 0.4, [description = "Static friction torque"]
        q̇_s = 0.1, [description = "Stribeck coefficient"]
        fsm_state = 0
        sign_velocity = 0
    end
    @variables begin
        q(t), [description = "Absolute rotation angle", guess = 0.0]
        q̇(t), [description = "Absolute angular velocity", guess = 0.0]
        # q̈(t), [description = "Absolute angular acceleration", guess = 0.0]
    end
    begin
        if stribeck
            str_scale = (τ_s - τ_c)
            β = exp(-abs(q̇/q̇_s))
            τ_f = (τ_c - str_scale * β) * sign_velocity + Kv * q̇
        else
            # τ_f = sign_velocity * τ_c + Kv * q̇
            τ_f = Kv * q̇
        end
    end
    @equations begin
        q ~ joint_in.q
        q ~ joint_out.q
        D(q) ~ q̇
        joint_out.τ ~ -joint_in.τ + τ_f
        # D(q̇) ~ q̈
    end
    # @continuous_events begin
    #     [joint_out.q̇ ~ 0] => (affect!, [joint_out.q̇], [τ_f.fsm_state, τ_f.sign_velocity], [], 0)
    # end
    # @discrete_events begin
    #     ((fsm_state == 0) * (joint_in.τ > τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 1)
    #     ((fsm_state == 0) * (joint_in.τ < -τ_s)) => (affect!, [], [fsm_state, sign_velocity], [], 2)
    # end
end

@mtkmodel GravitationalTorque begin
    @components begin
        joint_in = Joint()
        joint_out = Joint()
    end
    @variables begin
        q(t), [description = "Absolute rotation angle", guess = 0.0]
        q̇(t), [description = "Absolute angular velocity", guess = 0.0]
        # q̈(t), [description = "Absolute angular acceleration", guess = 0.0]
    end
    @equations begin
        q ~ joint_in.q
        q ~ joint_out.q
        D(q) ~ q̇
        joint_out.τ ~ joint_in.τ - compute_dynamics(mechanism, q, q̇)
        # D(q̇) ~ q̈
    end
end

@mtkmodel SP begin
    @structural_parameters begin
        experiment = 1
        stribeck = false
    end
    @components begin
        τ_input = OLController(kt = 0.2843, experiment = experiment)
        J_total = Inertia(J_motor = 0.07305)
        τ_f = FrictionTorque(stribeck = stribeck, τ_c = 0.128, Kv = 1.2465, τ_s = 0.128, q̇_s = 0.)
        τ_grav = GravitationalTorque()
    end
    @equations begin
        connect(τ_input.joint_out, τ_grav.joint_in)
        connect(τ_grav.joint_out, τ_f.joint_in)
        connect(τ_f.joint_out, J_total.joint_in)
    end
    # @parameters begin
    #     fsm_state = 0
    #     sign_velocity = 0
    # end
    # @continuous_events begin
    #     [J_total.q̇ ~ 0] => (affect!, [J_total.q̇ => :q̇], [fsm_state, sign_velocity], [], 0)
    # end
end

function build_problem(mechanism_; u0 = [0., 0.0], change_tspan = false, tspan = (0.0, 120.0), pars = [0.128, 1.2465, 0.2843, 0.07305, 0.128, 0], experiment=1, stribeck=false)
    global mechanism = mechanism_
    if !change_tspan
        tspan = (dfs[experiment].timestamp[1], dfs[experiment].timestamp[end])
    end
    if !stribeck
        push!(pars, 0.)
    end
    # @mtkbuild sp = SP(τ_input.experiment = experiment, τ_f.stribeck = stribeck)
    @mtkbuild sp = SP(experiment = experiment)

    # matrices, simplified_sys = linearize(sp, [sp.τ], [sp.θ])
    prob = ODEProblem(sp, [sp.τ_input.q => u0[1], sp.τ_input.q̇ => u0[2]], tspan, [sp.τ_f.τ_c => pars[1], sp.τ_f.Kv => pars[2], sp.τ_input.kt => pars[3], sp.J_total.J_motor => pars[4], sp.τ_f.τ_s => pars[5], sp.τ_f.q̇_s => pars[6]])
    # prob = ODEProblem(sp, [sp.τ_input.q => u0[1], sp.τ_input.q̇ => u0[2]], tspan, [sp.τ_input.kt => pars[3], sp.J_total.J_motor => pars[4]])

    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol, prob, sp
end

function plot_simulation(sol; experiment = 1)
    # println(sol.ps[-1])
    p1 = plot(sol.t, [sol.u[i][2] for i in 1:length(sol.t)], ylabel = "velocity, rad/s")
    p2 = plot(sol.t, [sol.u[i][1] for i in 1:length(sol.t)], ylabel = "position, rad")
    # torques = [compute_dynamics(mechanism, sol.u[i][1], sol.u[i][2]) + torque_ext(sol.t[i]) for i in 1:length(sol.t)]
    torques = [torque_ext(t, 0.2843; exp = experiment) for t in sol.t]
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

end