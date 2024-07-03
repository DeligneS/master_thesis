import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from constants import L1 as l1, L2 as l2

# Forward kinematics to get the Cartesian coordinates from the angles
def forward_kinematics(q1, q2):
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q1 + q2)
    y2 = y1 - l2 * np.cos(q1 + q2)
    return (x1, y1), (x2, y2)

def calculate_positions(q1, q2):
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q1 + q2)
    y2 = y1 - l2 * np.cos(q1 + q2)
    return (x1, y1), (x2, y2)

# Function to check if a point is within the rotated ellipsoid
def is_point_in_ellipsoid(q, mu, a, b, theta):
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    # Translate and rotate the point
    q_rotated = R @ (q - mu)
    # Check if the rotated point is inside the standard ellipsoid
    return (q_rotated[0] ** 2) / a ** 2 + (q_rotated[1] ** 2) / b ** 2 <= 1

# Generate random points and filter those within the rotated ellipsoid
def generate_ellipsoid_points(mu, a, b, theta, num_points=50000):
    points = []
    while len(points) < num_points:
        q1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        q2 = np.random.uniform(-np.pi / 2, np.pi / 2)
        q = np.array([q1, q2])
        if is_point_in_ellipsoid(q, mu, a, b, theta):
            points.append(q)
    return np.array(points)

# Function to check if a line segment intersects any ellipsoid
def check_collision_with_ellipsoids(x1, y1, x2, y2, ellipsoids):
    line = LineString([(x1, y1), (x2, y2)])
    for ellipsoid in ellipsoids:
        center_x, center_y = ellipsoid['center']
        a, b = ellipsoid['a'], ellipsoid['b']
        ellipse = Point(center_x, center_y).buffer(1)  # Create a circular buffer then scale it
        ellipse = scale(ellipse, xfact=a, yfact=b, origin=(center_x, center_y))
        if line.intersects(ellipse):
            return True
    return False


def check_collision(q1, q2, obstacle):
    (x1, y1), (x2, y2) = calculate_positions(q1, q2)
    arm1 = LineString([(0, 0), (x1, y1)])
    arm2 = LineString([(x1, y1), (x2, y2)])
    return arm1.intersects(obstacle) or arm2.intersects(obstacle)

def plot_state_space(points, cartesian_coords):
    # Plotting
    _, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Plot state space
    ax[0].scatter(points[:, 0], points[:, 1], color='red', s=1, label='Filled Collision States')
    ax[0].set_title('State Space (Filled Collision States)')
    ax[0].set_xlabel('$q_1$ (radians)')
    ax[0].set_ylabel('$q_2$ (radians)')
    ax[0].set_xlim(-np.pi / 2, np.pi / 2)
    ax[0].set_ylim(-np.pi / 2, np.pi / 2)
    ax[0].legend()

    # Plot in Cartesian space
    # Draw the pendulum in its initial state (no collision state)
    (q1, q2) = (0,0)  # Sample state
    (x1, y1), (x2, y2) = calculate_positions(q1, q2)
    ax[1].plot([0, x1, x2], [0, y1, y2], 'o-', label='Double Pendulum')

    x_coords, y_coords = cartesian_coords.T
    ax[1].scatter(x_coords, y_coords, color='blue', s=1, label='Obstacle Coverage')
    ax[1].set_xlim(-0.6, 0.6)
    ax[1].set_ylim(-0.5, 0.2)
    ax[1].set_title('Cartesian Space with Obstacle Coverage')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].set_aspect('equal')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
