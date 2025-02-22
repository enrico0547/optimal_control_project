import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation

from dynamics import dt

def animate_dynamic_system(xx_star, dt=dt, d=1.0):
    """""
    Animates the evolution of a dynamic system over time

    Args:
        - xx_star: State trajectory
        - dt: Time step duration
        - d: Distance between consecutive points along the horizontal axis (default: 1.0)
    """""
    TT = xx_star.shape[1]   # number of time steps
    p_positions = np.array([-d, 0, d, 2*d, 3*d, 4*d])   # positions including start and end
    z_fixed = np.zeros(2)   # heights for the fixed points at start and end

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5*d, 4.5*d)
    ax.set_ylim(-1.5 * np.max(xx_star), 1.5 * np.max(xx_star))

    # Dotted line at z = 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    aspect_ratio = (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0])

    base_triangle_size = 0.15
    triangle_size_x = base_triangle_size
    triangle_size_y = 2 * base_triangle_size / aspect_ratio

    # Add black triangles
    triangle_left = Polygon([[-d, 0], 
                         [-d - triangle_size_x, -triangle_size_y], 
                         [-d + triangle_size_x, -triangle_size_y]], 
                        closed=True, color="black")

    triangle_right = Polygon([[4*d, 0], 
                          [4*d - triangle_size_x, -triangle_size_y], 
                          [4*d + triangle_size_x, -triangle_size_y]], 
                         closed=True, color="black")
    ax.add_patch(triangle_left)
    ax.add_patch(triangle_right)

    # Plot elements
    system_line, = ax.plot([], [], 'o-', lw=2, color="blue", label="Optimal Path")
    z1, = ax.plot([], [], 'o--', lw=1, color="gray")
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.legend()
    ax.set_title("Dynamic System Animation")
    ax.set_xlabel("Position (p)")
    ax.set_ylabel("Height (z)")

    # Initial setup function for the animation
    def init():
        system_line.set_data([], [])
        z1.set_data([], [])
        time_text.set_text('')
        return system_line, z1, time_text

    # Update function for each frame of the animation
    def update(frame):
        """""
        Updates the positions of the dynamic points and the time text for the current frame.

        Args:
            - frame: Current frame index

        Returns:
            - Updated graphical elements for the animation
        """""
        # Positions and heights at the current time step
        p = p_positions
        z_dynamic = xx_star[:, frame]
        z = np.concatenate([[z_fixed[0]], z_dynamic, [z_fixed[1]]]) # add fixed points

        # Update system line
        system_line.set_data(p, z)

        # Update time text
        time_text.set_text(f'time = {frame * dt:.2f}s')

        return system_line, time_text

    # Create the animation
    _ = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=dt * 1000)

    # Display the animation
    plt.show()