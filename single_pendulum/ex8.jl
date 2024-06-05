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
data = [[pi/4, pi/3, 0.0], [pi/6, pi/2, 0.1], [pi/3, pi/4, 0.2]]

# Pendulum arm lengths
l1, l2 = 1.0, 1.0

# Initialize plot
p0 = plot(size=(800, 600), legend=false)

# Iterate through data to plot the pendulum at each time step
for (q1, q2, t1) in data
    x1, y1, x2, y2 = calculate_positions(q1, q2, l1, l2)
    plot!(p0, [0, x1, x2], [0, y1, y2], label="", marker=:circle)
end

plot(p0, layout=(1,1), link=:x, leg = false)