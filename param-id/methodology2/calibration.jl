using JuliaSimModelOptimizer

include("../../URDF/model_urdf.jl")
# # include("model_st.jl")
include("model_st_tanh.jl")
# # mechanism = get_mechanism(;wanted_mech = "single_pendulum")
mechanism = get_mechanism(;wanted_mech = "single_pendulum_knee")

exps = 4
exps1 = 15
exps2 = 16
exps2 = 22
exps2 = 17

sys1 = modeler(mechanism; experiment = exps1)
sys2 = modeler(mechanism; experiment = exps2)

# r = [0.128, 1.2465, 0.2843, 0.07305]
# sol = simulation(sys2; u0 = [0., 0.0], tspan = (0.0, 200.0), pars = r)
# plot_simulation(sol, dfs[exps2]; experiment = exps2)

# trial1 = Experiment(df1, sys1, save_names = ["θ(t)"])
# trial2 = Experiment(df2, sys2, save_names = ["θ(t)"])
# trial1 = Experiment(dfs[exps1], sys1, save_names = ["θ(t)"], alg = Tsit5(), model_transformations = [PredictionErrorMethod(0.05)])
# trial2 = Experiment(dfs[exps2], sys2, save_names = ["θ(t)", "ω(t)"], alg = Tsit5(), model_transformations = [PredictionErrorMethod(0.05)])
trial2 = Experiment(dfs[exps2], sys2, save_names = ["θ(t)"], alg = Tsit5(), model_transformations = [PredictionErrorMethod(0.05)])

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
# alg = SingleShooting(maxiters = 30, optimizer = IpoptOptimizer(tol = 1e-5, acceptable_tol = 1e-4))
alg = SingleShooting(maxtime = 10, optimizer = IpoptOptimizer(tol = 1e-5, acceptable_tol = 1e-4))

r = calibrate(invprob, alg, progress = true)
println(r)
# [0.03825593847663786, 1.200371929157816, 0.33872056672315987, 0.047237338208489396, 0.1505977042688268]
# [0.05908656478893748, 1.238429215802329, 0.3557984214111448, 0.045223242064243954, 0.0970333011883011, 0.02406968918165099]
