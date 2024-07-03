import numpy as np
import matplotlib.pyplot as plt
from utils import forward_kinematics, calculate_positions, generate_ellipsoid_points
from constants import L1, L2
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})

def initialize_ellipsoid():
    mu = np.array([0.0, 0.0])  # Center of the ellipsoid
    factor = 0.6
    a, b = 0.41 / 2, 2.82 / 2 # Semi-axes lengths
    a, b = a * factor, b * factor # Tune the size of the ellipsoid
    theta = -25.537 * np.pi/180  # Rotation angle in radians
    return mu, a, b, theta

def generate_ellipsoid_data(mu, a, b, theta):
    points = generate_ellipsoid_points(mu, a, b, theta)
    cartesian_coords = np.array([forward_kinematics(p[0], p[1])[1] for p in points])
    return points, cartesian_coords

def plot_state_space(ax, points, initial_state, final_state, mu, a, b, theta):
    ax.scatter(points[:, 0], points[:, 1], color='indianred', s=3)
    ax.scatter(initial_state[0], initial_state[1], color='darkgreen', label='Initial State')
    ax.scatter(final_state[0], final_state[1], color='cornflowerblue', label='Goal State')

    # Generate ellipse boundary points
    t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)
    # Rotate the points
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    ellipse_points = np.dot(R, np.array([ellipse_x, ellipse_y]))
    # Translate the points
    ellipse_points[0, :] += mu[0]
    ellipse_points[1, :] += mu[1]

    # Plot ellipse outline
    ax.plot(ellipse_points[0, :], ellipse_points[1, :], color = 'firebrick', label='Ellipsoid Collision States')


    ax.set_title('State Space')
    ax.set_xlabel('$q_1$ (radians)')
    ax.set_ylabel('$q_2$\n(radians)', labelpad=20, rotation=0)
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    ax.grid()

def plot_cartesian_space(ax, initial_state, final_state, cartesian_coords):
    q1, q2 = initial_state
    q1f, q2f = final_state
    (x1, y1), (x2, y2) = calculate_positions(q1, q2)
    (x1f, y1f), (x2f, y2f) = calculate_positions(q1f, q2f)
    ax.plot([0, x1, x2], [0, y1, y2], 'o-', color = 'darkgreen', label='Double Pendulum Initial Position')
    ax.plot([0, x1f, x2f], [0, y1f, y2f], 'o-', color = 'cornflowerblue', label='Double Pendulum Goal Position')
    x_coords, y_coords = cartesian_coords.T
    ax.scatter(x_coords, y_coords, color='firebrick', s=1, label='Obstacle Coverage')
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.65, 0.2)
    ax.set_title('Cartesian Space with Obstacle Coverage')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]', labelpad=20, rotation=0)
    ax.set_aspect('equal')
    ax.grid()

def calc_segment(thetas):
    segments = []
    for theta in thetas :
        theta1, theta2_offset = theta
        segments.append((L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2_offset),
                L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2_offset)))
    return segments

def main():
    mu, a, b, theta = initialize_ellipsoid()
    points, cartesian_coords = generate_ellipsoid_data(mu, a, b, theta)

    initial_state = (-np.pi/10, -np.pi/10)
    final_state = (np.pi/10, np.pi/8)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    plot_state_space(ax[0], points, initial_state, final_state, mu, a, b, theta)
    plot_cartesian_space(ax[1], initial_state, final_state, cartesian_coords)
    
    # Define theta ranges and offsets for the reachable set
    theta_ranges = [
        (np.linspace(-np.pi/2, np.pi/2, 100), 0),
        (np.pi/2, np.linspace(0, np.pi/2, 100)),
        (np.linspace(np.pi/2, -np.pi/2, 100), np.pi/2),
        (np.linspace(0, -np.pi/2, 100), -np.pi/2),
        (-np.pi/2, -np.linspace(0, np.pi/2, 100))
    ]
    segments = calc_segment(theta_ranges)
    # for seg in segments:
    #     ax[1].plot(-seg[1], -seg[0], 'b')  # Plots all segments
    ax[1].fill(-np.concatenate([seg[1] for seg in segments]),
               -np.concatenate([seg[0] for seg in segments]), 'grey', alpha=0.3, label="End-Effector's Reachable Set Area")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
