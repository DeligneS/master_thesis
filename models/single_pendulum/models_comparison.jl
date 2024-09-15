using CSV, DataFrames, DifferentialEquations
include(joinpath("..", "model_urdf.jl"))
include("model_callback.jl")
include("single_pendulum_system.jl")

mechanism = get_mechanism(; wanted_mech = "single_pendulum")

# Validation set : 9 -> 14
experiment = 10
# r = [0.14143497938317678, 1.2785059304172441, 0.39382068682857385, 0.048592330795521735, 0.05246035748938784, 0.014025757542366207] # 1h
# r = [0.11741464750529594, 1.2380507798987224, 0.38065022439819174, 0.11755681827219377, 0.12876092204679299, 0.029231003865808534] # 2min
# r = [0.0706396353290064, 1.5651123281135342, 0.3952914552970312, 0.040786221279949124, 0.19756590009983593, 0.009441932814455819] # 1h on non-trivial
# r = [0.0706396353290064, 0.0706396353290064, 1.5651123281135342, 0.3952914552970312, 0.040786221279949124] # 1h on non-trivial

folder = "models4"
for i in 1:6
    experiment = i + 8
    # Model N°1 : paramètres via méthodologie 1
    r = [0.128, 1.2465, 0.2843, 0.07305, 0.128, 0] # [τ_c, Kv, kt, J_motor, τ_s, q̇_s]
    sys, model = single_pendulum_system.system(mechanism; pars = r, experiment = experiment)
    prob = single_pendulum_system.problem(sys, experiment = experiment)
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    df = DataFrame(Time = sol.t, Position = [u[1] for u in sol.u])
    file_name = "$folder/simu_model1_exp$i.csv"
    CSV.write(file_name, df)

    # Model N°2 : model 1, avec le kt manual tuned
    facteur_correctif = 1.15
    r = [0.128, 1.2465, 0.2843 * facteur_correctif, 0.07305, 0.128, 0]
    sys, model = single_pendulum_system.system(mechanism; pars = r, experiment = experiment)
    prob = single_pendulum_system.problem(sys, experiment = experiment)
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    df = DataFrame(Time = sol.t, Position = [u[1] for u in sol.u])
    file_name = "$folder/simu_model2_exp$i.csv"
    CSV.write(file_name, df)

    # Model N°3 : paramètres via méthodologie 2, tanh
    sys, model = single_pendulum_system.system(mechanism; experiment = experiment)
    prob = single_pendulum_system.problem(sys, experiment = experiment)
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    df = DataFrame(Time = sol.t, Position = [u[1] for u in sol.u])
    file_name = "$folder/simu_model3_exp$i.csv"
    CSV.write(file_name, df)

    # Model N°4 : paramètres via méthodologie 2, Stribeck effect, tanh
    r = [0.06546871046131894, 1.585098291931009, 0.39550177534910586, 3.801747998965773e-5, 0.04642699663443199, 0.009102450216353445]
    sys, model = single_pendulum_system.system(mechanism; pars = r, stribeck = true, experiment = experiment)
    prob = single_pendulum_system.problem(sys, experiment = experiment)
    sol = solve(prob, Tsit5(), dtmax=0.01, reltol=1e-5, abstol=1e-5)
    df = DataFrame(Time = sol.t, Position = [u[1] for u in sol.u])
    file_name = "$folder/simu_model4_exp$i.csv"
    CSV.write(file_name, df)
end

# Model N°3 : paramètres via méthodologie 2, tanh
# sol, prob, sys = model_tanh.build_problem(mechanism, state; experiment = experiment)

# Model N°4 : paramètres via méthodologie 2, callbacks
# r = [0.07, 0.0706396353290064, 1.5651123281135342, 0.3952914552970312, 0.040786221279949124]
# sol, prob, sys = model_callback.build_problem(mechanism, state; experiment = experiment)
# sol, prob, sys = single_pendulum_model.build_problem(mechanism, state; experiment = experiment)


# single_pendulum_model.plot_simulation(sol; experiment = experiment)
# model_callback.plot_simu_vs_real(sol; experiment = experiment)
