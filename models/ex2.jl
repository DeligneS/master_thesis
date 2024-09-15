include("../URDF/model_urdf.jl")
include("single_pendulum/model_callback.jl")

mechanism = get_mechanism(;wanted_mech = "single_pendulum")

experiment = 17

r = [0.06510461345450586, 1.5879662676966781, 0.39454422423683916, 0., 0.06510461345450586]
sol, prob, sys = model_callback.build_problem(mechanism; experiment = experiment)

p0 = plot(sol.t, sol[sys.q̈], ylabel="    Torque [Nm]", label="Torque", color="green", legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

plot(p0, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
