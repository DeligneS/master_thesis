using Plots

# Function to calculate Cartesian coordinates from angles
function calculate_positions(q1, q2, l1, l2)
    x1 = l1 * sin(q1)
    y1 = -l1 * cos(q1)
    x2 = x1 + l2 * sin(q2)
    y2 = y1 - l2 * cos(q2)
    return x1, y1, x2, y2
end

# Sample input data
data = [[pi/4, pi/3, 0.0], [pi/6, pi/2, 0.1], [pi/3, pi/4, 0.2], [pi/2, pi/3, 0.3], [pi/4, pi/6, 0.4]]

# Pendulum arm lengths
l1, l2 = 1.0, 1.0

# Sort data by time to ensure animation goes in chronological order
sorted_data = sort(data, by = x -> x[3])

# Initialize trace for end-effector
trace_x, trace_y = [], []

# Generate animation
anim = @animate for (q1, q2, t1) in sorted_data
    x1, y1, x2, y2 = calculate_positions(q1, q2, l1, l2)
    
    # Update the trace of the end-effector
    push!(trace_x, x2)
    push!(trace_y, y2)
    
    # Clear current plot and redraw for smooth animation
    plot([0, x1, x2], [0, y1, y2], xlims=(-2.5, 2.5), ylims=(-2.5, 0.5),
         marker=:circle, markersize=6, color=:blue, legend=false,
         title="Time: $(round(t1, digits=2)) seconds")
    
    # Add the trace of the end-effector
    plot!(trace_x, trace_y, line=(2, :red))
end

# Save the animation as a GIF file
gif(anim, "double_pendulum_end_effector_trace.gif", fps = 2)
