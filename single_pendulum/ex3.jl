using ModelingToolkit
using ModelingToolkit: t_nounits as t
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Mechanical.Rotational
using ModelingToolkitStandardLibrary.Blocks
using OrdinaryDiffEq
using Plots

R = 0.5 # [Ohm] armature resistance
L = 4.5e-3 # [H] armature inductance
k = 0.5 # [N.m/A] motor constant
J = 0.02 # [kg.m²] inertia
f = 0.01 # [N.m.s/rad] friction factor
tau_L_step = -0.3 # [N.m] amplitude of the load torque step

@named ground = Ground()
@named source = Voltage()
@named ref = Blocks.Step(height = 1, start_time = 0)
@named pi_controller = Blocks.LimPI(k = 1.1, T = 0.035, u_max = 10, Ta = 0.035)
@named feedback = Blocks.Feedback()
@named R1 = Resistor(R = R)
@named L1 = Inductor(L = L)
@named emf = EMF(k = k)
@named fixed = Fixed()
@named load = Torque()
@named load_step = Blocks.Step(height = tau_L_step, start_time = 3)
@named inertia = Inertia(J = J)
@named friction = Damper(d = f)
@named speed_sensor = SpeedSensor()

connections = [connect(fixed.flange, emf.support, friction.flange_b)
               connect(emf.flange, friction.flange_a, inertia.flange_a)
               connect(inertia.flange_b, load.flange)
               connect(inertia.flange_b, speed_sensor.flange)
               connect(load_step.output, load.tau)
               connect(ref.output, :r, feedback.input1)
               connect(speed_sensor.w, :y, feedback.input2)
               connect(feedback.output, pi_controller.err_input)
               connect(pi_controller.ctr_output, :u, source.V)
               connect(source.p, R1.p)
               connect(R1.n, L1.p)
               connect(L1.n, emf.p)
               connect(emf.n, source.n, ground.g)]

@named model = ODESystem(connections, t,
    systems = [
        ground,
        ref,
        pi_controller,
        feedback,
        source,
        R1,
        L1,
        emf,
        fixed,
        load,
        load_step,
        inertia,
        friction,
        speed_sensor
    ])

sys = structural_simplify(model)
prob = ODEProblem(sys, [], (0, 6.0))

using BenchmarkTools
@btime sol = solve(prob, Rodas4())

p1 = Plots.plot(sol.t, sol[inertia.w], ylabel = "Angular Vel. in rad/s",
    label = "Measurement", title = "DC Motor with Speed Controller")
Plots.plot!(sol.t, sol[ref.output.u], label = "Reference")
p2 = Plots.plot(sol.t, sol[load.tau.u], ylabel = "Disturbance in Nm", label = "")
Plots.plot(p1, p2, layout = (2, 1))


matrices, simplified_sys = linearize(model, :r, :y)


# Named State Space
using ControlSystemsMTK

lsys = named_ss(model, :r, :y)

w = exp10.(LinRange(-0.5, 4, 200))
ControlSystemsMTK.bodeplot(lsys, w)

# Compute sensitivity in analysis
using ControlSystemsBase
matrices_S, simplified_sys = Blocks.get_sensitivity(model, :y)
So = ss(matrices_S...) |> minreal # The output-sensitivity function as a StateSpace system
matrices_T, simplified_sys = Blocks.get_comp_sensitivity(model, :y)
To = ss(matrices_T...)# The output complementary sensitivity function as a StateSpace system
bodeplot([So, To], label = ["S" "T"], plot_title = "Sensitivity functions",
    plotphase = false)


matrices_L, simplified_sys = Blocks.get_looptransfer(model, :y)
L = -ss(matrices_L...) # The loop-transfer function as a StateSpace system. The negative sign is to negate the built-in negative feedback
Ms, ωMs = hinfnorm(So) # Compute the peak of the sensitivity function to draw a circle in the Nyquist plot
nyquistplot(L, label = "\$L(s)\$", ylims = (-2.5, 0.5), xlims = (-1.2, 0.1),
    Ms_circles = Ms)
    