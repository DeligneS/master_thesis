using Plots

function plot_estimation_vs_exp(sol, df)
    p2 = plot(sol.t, [u[1] for u in sol.u], ylabel="    Position [rad]", label="Simulation", color="blue", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p2, df.t, df.DXL_Position, color="red", label="Real data", linewidth=2)

    plot(p2, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
end

function plot_estimation_vs_real(sol, sol2)
    p2 = plot(sol.t, [u[1] for u in sol.u], ylabel="    Position [rad]", label="real", color="blue", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p2, sol2.t, [u[1] for u in sol2.u], ylabel="    Position [rad]", label="estimation", color="red", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

    p0 = plot(sol.t, [u[2] for u in sol.u], ylabel="    Velocity [rad/s]", label="real", color="blue", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)
    plot!(p0, sol2.t, [u[2] for u in sol2.u], ylabel="    Velocity [rad/s]", label="estimation", color="red", 
            legendfontsize=12, tickfontsize=10, yguidefontrotation=0, linewidth=2)

    plot(p0, p2, layout=(2, 1), link=:x, size=(800, 600), xlabel="Time [s]")
end