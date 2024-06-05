using JuliaSimModelOptimizer
using OrdinaryDiffEq
using CSV, DataFrames
using IfElse: ifelse
using Statistics
using StatsPlots
# import DataInterpolations: CubicSpline
using ModelingToolkit

using DataSets
using Plots

include("../../last_mtk_version/model_urdf.jl")
# include("model_st.jl")
include("model_st_tanh.jl")
mechanism = get_mechanism(;wanted_mech = "single_pendulum")

exps = 4
# exps = (8, df8)
# exps = (9, df9)
# exps = (10, df10)
# exps = (11, df11)
# exps = (12, df12)
# exps = (13, df13)
# exps = (14, df14)
exps1 = 15
exps2 = 16
sys1 = modeler(mechanism; experiment = exps1)
sys2 = modeler(mechanism; experiment = exps2)

# r = [0.128, 1.2465, 0.2843, 0.07305]
# sol = simulation(sys2; u0 = [0., 0.0], tspan = (0.0, 200.0), pars = r)
# plot_simulation(sol, dfs[exps2]; experiment = exps2)

# trial1 = Experiment(df1, sys1, save_names = ["θ(t)"])
# trial2 = Experiment(df2, sys2, save_names = ["θ(t)"])
# trial1 = Experiment(dfs[exps1], sys1, save_names = ["θ(t)"], alg = Tsit5(), model_transformations = [PredictionErrorMethod(0.05)])
trial2 = Experiment(dfs[exps2], sys2, save_names = ["θ(t)", "ω(t)"], alg = Tsit5(), model_transformations = [PredictionErrorMethod(0.05)])

# invprob = InverseProblem([trial2], sys2,
#     [
#         sys2.τ_c => (0., 0.2),
#         sys2.Kv => (0., 2.),
#         sys2.kt => (0., 1.),
#         # sys2.τ_s => (0., 0.4),
#         sys2.str_scale => (0., 0.3),
#         sys2.q̇_s => (0., 0.1),
#         sys2.J_motor => (0., 0.1)
#     ]
# )
invprob = InverseProblem([trial2], sys2,
    [
        sys2.τ_c => (0., 0.2),
        sys2.Kv => (0., 2.),
        sys2.kt => (0., 1.),
        sys2.J_motor => (0., 0.1)
    ]
)
# par = @parameters τ_s = 0.128 τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305
# par = @parameters τ_c = 0.128 Kv = 1.2465 kt = 0.2843 J_motor = 0.07305 τ_brk = 0.128 w_brk = 0.1 fsm_state = 1 sign_velocity = 1
# [0.14143497938317678, 1.2785059304172441, 0.39382068682857385, 0.048592330795521735, 0.05246035748938784, 0.014025757542366207]

# # ps = parametric_uq(invprob, StochGlobalOpt(maxiters = 100), sample_size = 50)
alg = SingleShooting(maxiters = 30, optimizer = IpoptOptimizer(tol = 1e-5, acceptable_tol = 1e-4))
# alg = SingleShooting(maxtime = 10, optimizer = IpoptOptimizer(tol = 1e-5, acceptable_tol = 1e-4))

r = calibrate(invprob, alg, progress = true)
println(r)
# [0.03825593847663786, 1.200371929157816, 0.33872056672315987, 0.047237338208489396, 0.1505977042688268]
# [0.05908656478893748, 1.238429215802329, 0.3557984214111448, 0.045223242064243954, 0.0970333011883011, 0.02406968918165099]

# plot(trial1, invprob, r, show_data = true, layout = (1, 1), size = (1000, 600), ms = 0.1)

# include("plotting.jl")
# # sol1, prob1, sys1 = modeler(mechanism, state, shoulder;u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref = torque_ref, tau_s = 0.128, tau_c = 0.128, K_v = 1.2465, k_t = 0.2843)
# # sol2, prob2, sys2 = modeler(mechanism, state, shoulder;u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref = torque_ref, tau_s = 0.175, tau_c = 0.175, K_v = 1.89113, k_t = 0.254534)
# sol2, prob2, sys2 = modeler(mechanism, state, shoulder;u0 = [0., 0.0], tspan = (0.0, 120.0), torque_ref = torque_ref, tau_s = r[1], tau_c = r[2], K_v = r[3], k_t = r[4])
# plot_estimation_vs_real(sol, sol2)

# p1 = density(params[:, 1], label = "Estimate: tau_s")
# plot!([0.128 , 0.128 ], [0.0, 3],
#       lw=3, color=:green, label="True value: tau_s", linestyle= :dash)

# p2 = density(params[:, 2], label = "Estimate: tau_c")
# plot!([0.128 , 0.128 ], [0.0, 1.5],
#     lw=3, color=:red, label="True value: tau_c", linestyle= :dash)

# p3 = density(params[:, 3], label = "Estimate: Kv")
# plot!([1.246491820256837, 1.246491820256837], [0.0, 0.15],
#         lw=3, color=:purple, label="True value: Kv", linestyle= :dash)

# p4 = density(params[:, 4], label = "Estimate: kt")
# plot!([0.28432313665258757, 0.28432313665258757], [0.0, 3],
#         lw=3, color=:purple, label="True value: kt", linestyle= :dash)

# p5 = density(params[:, 5], label = "Estimate: Jm")
# plot!([0.07305, 0.07305], [0.0, 30],
#         lw=3, color=:purple, label="True value: Jm", linestyle= :dash)

# l = @layout [a b c d e]
# plot(p1, p2, p3, p4, p5, layout = l)