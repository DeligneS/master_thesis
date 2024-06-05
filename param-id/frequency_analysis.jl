using JuliaSimModelOptimizer
using ControlSystemIdentification
using OrdinaryDiffEq
using CSV, DataFrames
using IfElse: ifelse
using Statistics
using StatsPlots
import DataInterpolations: CubicSpline
using ModelingToolkit

using DataSets
using Plots

include("../single_pendulum/single_pendulum_urdf.jl")
include("model_st.jl")
mechanism, state, shoulder = single_pendulum.single_pendulum_mechanism()

exps = (4, df4)
exps = (8, df8)
exps = (9, df9)
# exps = (10, df10)
# exps = (11, df11)
# exps = (12, df12)
# exps = (13, df13)
# exps = (14, df14)
exps = (15, df15)
# exps = (16, df16)

r = [0.14143497938317678, 1.2785059304172441, 0.39382068682857385, 0.048592330795521735, 0.05246035748938784, 0.014025757542366207] # 1h
r = [0.11741464750529594, 1.2380507798987224, 0.38065022439819174, 0.11755681827219377, 0.12876092204679299, 0.029231003865808534] # 2min
r = [0.0706396353290064, 1.5651123281135342, 0.3952914552970312, 0.040786221279949124, 0.19756590009983593, 0.009441932814455819] # 1h on non-trivial

sys = modeler(mechanism, state; experiment = exps[1])

sol = simulation(sys; u0 = [0., 0.0], tspan = (0.0, 120.0), pars = r)
# plot_simulation(sol, exps[2]; experiment = exps[1])

u_list = map(u_functions[exps[1]], sol.t)
u_list = map(u_functions[exps[1]], exps[2].timestamp)

fig1 = let
    # simulation = iddata(sol, u_list, sol.t[13] - sol.t[12])
    simulation = iddata(sol, u_list, sol.t[2] - sol.t[1])
    simulation = iddata(exps[2][!, "θ(t)"], u_list, 0.02)
    # simulation.t = sol.t
	figdata = plot(simulation, plot_title="Chirp")

	# fig1 = plot(
	# 	figdata,
	# 	welchplot(simulation, title="Spectra for \$P_2(s)\$ experiment"),
	# )
	# fig2 = specplot(simulation)
    fig2 = welchplot(simulation, title="Spectra for \$P_2(s)\$ experiment")
	fig2
end;

