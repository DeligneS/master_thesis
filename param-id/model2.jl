using ModelingToolkit, DifferentialEquations, LinearAlgebra, Plots, IfElse, StructuralIdentifiability, DataFrames, RigidBodyDynamics, CSV, DataFrames

# include("single_pendulum_ode_simplify.jl")
include("../../utils/torque_references/references.jl")
# include("../../utils/torque_references/discrete_references.jl")

# Using the module or its functions
using .torque_references: waveform_value_at_time
using .torque_references: current_for_time
# using .discrete_references: get_u_at_time
struct UFunction
    df::DataFrame
end
df1 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df2 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df3 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df4 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df5 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)
# df6 = CSV.read("/Users/simondeligne/Dionysos/dev_perso/Dionysos.jl/obstacle_avoidance_simon/utils/recorded_data/chirp/M1_chirp_05_200.csv", DataFrame)

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

u_function = UFunction(df1)

# External torque function (as before, consider any changes if necessary)
function torque_ext(t, kt; ref = "triangle")
    # voltage = waveform_value_at_time(t, ref)
    voltage = u_function(t)
    # voltage = current_for_time(t)
    return voltage * kt
end

function modeler(mechanism, state, shoulder;u0 = [0., 0.0], tspan = (0.0, 60.0), torque_ref = "triangle", tau_s = 0.0128, tau_c = 0.0128, K_v = 1.2465, k_t = 0.2843)
    
    function compute_dynamics(mechanism, θ, ω)
        actual_state = MechanismState(mechanism, [θ], [ω])
        v̇ = copy(velocity(state))
        v̇[1] = 0
        τ_dynamics = -inverse_dynamics(actual_state, v̇)[1]
        return τ_dynamics
    end

    ##### Definition of the model in MTK V8.73 #####
    function affect!(integ, u, p, ctx)
        if (ctx == 0) # STATIC
            integ.p[p.fsm_state] = 0
            integ.u[u.ω] = 0.
        elseif (ctx == 1) # MOVING_CCW
            integ.p[p.fsm_state] = 1
            integ.p[p.sign_velocity] = 1
        elseif (ctx == 2) # MOVING_CW
            integ.p[p.fsm_state] = 1
            integ.p[p.sign_velocity] = -1
        end
    end

    isign(x) = IfElse.ifelse(signbit(x), -1, 1)
    @variables t
    sts = @variables τ(t) = 0. θ(t) = 0. ω(t) = 0.
    D = Differential(t)
    par = @parameters τ_s = 0.128 τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305 fsm_state = 1 sign_velocity = 1
    eqs = [
        τ ~ torque_ext(t, kt; ref = torque_ref)
        D(θ) ~ ω
        D(ω) ~ fsm_state * (compute_dynamics(mechanism, θ, ω) + τ - isign(ω) * τ_c - Kv * ω) / (J_motor + mass_matrix(MechanismState(mechanism, [θ], [ω]))[1])
    ]
    continuous_event = [ω ~ 0] => (affect!, [ω], [fsm_state, sign_velocity], 0)
    discrete_event = [((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = torque_ref)) > τ_s)) => (affect!, [], [fsm_state, sign_velocity], 1)
                      ((fsm_state == 0) * ((compute_dynamics(mechanism, θ, ω) + torque_ext(t, kt; ref = torque_ref)) < -τ_s)) => (affect!, [], [fsm_state, sign_velocity], 2)
                    #   ((fsm_state == 0) * ) 
                      ]
    @named sys = ODESystem(eqs, t, sts, par, continuous_events = continuous_event, discrete_events = discrete_event)
    sys = structural_simplify(sys)

    prob = ODEProblem(sys, [sys.θ => u0[1], sys.ω => u0[2]], tspan, [sys.τ_s => tau_s, sys.τ_c => tau_c, sys.Kv => K_v, sys.kt => k_t])

    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    return sol, prob, sys
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
