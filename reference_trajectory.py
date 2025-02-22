import numpy as np
import scipy as sp
import dynamics as dyn
import equilibria as eq

import matplotlib.pyplot as plt

# Parameters
d = dyn.d
alpha = dyn.alpha
m = dyn.m
m_act = dyn.m_act
c = dyn.c
x0=eq.x0
dt=dyn.dt
ns=dyn.ns
ni=dyn.ni

def gen(tf, u_initial, u_final, step=True):
    """""
    Calculates equilibrium points and generates a reference trajectory for states and inputs of a dynamic system.

    Args:
        tf: Total time duration of the trajectory
        u_initial: Initial input vector
        u_final: Final input vector
        step: Boolean flag to generate a step or a smooth reference between 2 equilibrium points (default: True)
        
    Returns:
        - xx_ref: State reference trajectory with shape
        - uu_ref: Input reference trajectory with shape
    """""
    xx_eq_initial = eq.equilibrium_points(u_initial).copy()
    xx_eq_final = eq.equilibrium_points(u_final).copy()

    print("Initial equilibrim ", xx_eq_initial)
    print("Initial equilibrim with fsolve ", sp.optimize.fsolve(eq.equilibrium_function, x0, args=(u_initial,)))
    print("Final equilibrim ", xx_eq_initial)
    print("Final equilibrim with fsolve ", sp.optimize.fsolve(eq.equilibrium_function, x0, args=(u_final,)))

    initial_state = np.array([[xx_eq_initial[0]], [xx_eq_initial[1]], [xx_eq_initial[2]], [xx_eq_initial[3]], [0.0], [0.0], [0.0], [0.0]])
    initial_input = np.array([[u_initial[1]], [u_initial[3]]])

    final_state = np.array([[xx_eq_final[0]], [xx_eq_final[1]], [xx_eq_final[2]], [xx_eq_final[3]], [0.0], [0.0], [0.0], [0.0]])
    final_input = np.array([[u_final[1]], [u_final[3]]])


    TT = int(tf / dt)
    flat_time_ratio = 0.3   # ratio of flat time to total time (30% at the start and end)
    flat_TT = int(flat_time_ratio * TT) # number of time steps for the flat parts
    transition_TT = TT - 2 * flat_TT    # time steps for the transition part

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    if not step:
        # Flat part (start and end)
        xx_ref[:, :flat_TT] = initial_state
        xx_ref[:, -flat_TT:] = final_state
        
        uu_ref[:, :flat_TT] = initial_input
        uu_ref[:, -flat_TT:] = final_input

        # Transition part (smooth trajectory)
        t_transition = np.linspace(0, 1, transition_TT) # normalized time vector
        smooth_curve = 0.5 * (1 - np.cos(np.pi * t_transition)) # scaled cosine curve for smoothness

        for i in range(4):  # for the first four state components (positions)
            xx_ref[i, flat_TT:-flat_TT] = (initial_state[i] + smooth_curve * (final_state[i] - initial_state[i]))

        # Consider a quasi-static trajectory 
        xx_ref[4:8, :] = 0

        # Update control inputs based on dynamics
        for i in range(flat_TT, TT - flat_TT):
            uu_ref[0, i] = alpha * dyn.coupling_sum(xx_ref[:, i], 1) + c * xx_ref[5, i]
            uu_ref[1, i] = alpha * dyn.coupling_sum(xx_ref[:, i], 3) + c * xx_ref[7, i]

    else:
        # Step reference (not modified)
        half_TT = int(TT / 2)
        xx_ref[:4, :half_TT] = np.tile(initial_state[:4], (1, half_TT)) # initial positions
        xx_ref[:4, half_TT:] = np.tile(final_state[:4], (1, TT - half_TT))  # final positions
        # Note: here the velocity is not updated, so remains zero (the whole vector xx_ref is initialized before)

        uu_ref[0, :half_TT] = alpha * dyn.coupling_sum(xx_ref[:4, :half_TT], 1)
        uu_ref[0, half_TT:] = alpha * dyn.coupling_sum(xx_ref[:4, half_TT:], 1)

        uu_ref[1, :half_TT] = alpha * dyn.coupling_sum(xx_ref[:4, :half_TT], 3)
        uu_ref[1, half_TT:] = alpha * dyn.coupling_sum(xx_ref[:4, half_TT:], 3)

    # PLOT

    fig, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)  # adjust figsize for better vertical space
    fig.canvas.manager.set_window_title('Reference')

    tt_hor = np.linspace(0,tf,TT)

    # Plotting each axis
    labels = ['$z_1$', '$z_2$', '$z_3$', '$z_4$', r'$\dot{z}_1$', r'$\dot{z}_2$', r'$\dot{z}_3$', r'$\dot{z}_4$', '$u_1$', '$u_2$']

    for i, ax in enumerate(axs):
        if i < 8:
            ax.plot(tt_hor, xx_ref[i, :], 'g--', linewidth=2)
            # ax.set_ylim([-1.5*max(xx_ref[i, :]), 1.5*max(xx_ref[i, :])])
        else:
            ax.plot(tt_hor, uu_ref[i - 8, :], 'r--', linewidth=2)
            # ax.set_ylim([-1.5*min(uu_ref[0, :], uu_ref[1, :]), 1.5*max(uu_ref[0, :], uu_ref[1, :])])
        ax.grid()
        ax.set_xlim([0, tf])
        ax.set_ylabel(labels[i], fontsize=15)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)    # reduce tick label size
    axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

    fig.subplots_adjust(hspace=0.08)    # adjust vertical spacing
    plt.tight_layout()

    plt.show()

    return xx_ref, uu_ref


def gen_swing(tf, u0, u1, u2, u3):
    """""
    Calculates equilibrium points and generates a reference trajectory for a dynamic system with transitions between states.

    Args:
        tf: Total time duration of the trajectory
        u0: Initial input vector
        u1: Intermediate input vector
        u2: Intermediate input vector
        u3: Final input vector
    
    Returns:
        - xx_ref: State reference trajectory.
        - uu_ref: Input reference trajectory.
    """""
    xx_eq_0 = eq.equilibrium_points(u0).copy()
    xx_eq_1 = eq.equilibrium_points(u1).copy()
    xx_eq_2 = eq.equilibrium_points(u2).copy()
    xx_eq_3 = eq.equilibrium_points(u3).copy()

    print("eq0 ", xx_eq_0)
    print("eq1 ", xx_eq_1)
    print("eq2 ", xx_eq_2)
    print("eq3 ", xx_eq_3)

    state_0 = np.array([[xx_eq_0[0]], [xx_eq_0[1]], [xx_eq_0[2]], [xx_eq_0[3]], [0.0], [0.0], [0.0], [0.0]])
    input_0 = np.array([[u0[1]], [u0[3]]])

    state_1 = np.array([[xx_eq_1[0]], [xx_eq_1[1]], [xx_eq_1[2]], [xx_eq_1[3]], [0.0], [0.0], [0.0], [0.0]])
    input_1 = np.array([[u1[1]], [u1[3]]])

    state_2 = np.array([[xx_eq_2[0]], [xx_eq_2[1]], [xx_eq_2[2]], [xx_eq_2[3]], [0.0], [0.0], [0.0], [0.0]])
    input_2 = np.array([[u2[1]], [u2[3]]])

    state_3 = np.array([[xx_eq_3[0]], [xx_eq_3[1]], [xx_eq_3[2]], [xx_eq_3[3]], [0.0], [0.0], [0.0], [0.0]])
    input_3 = np.array([[u3[1]], [u3[3]]])

    TT = int(tf / dt)
    n_segments = 7  # to divide the trajectory in 7 equal temporal portions
    TT_seg = int(TT / n_segments)   # number of time steps for the flat parts

    ti1 = TT_seg        # initial time for the first transition
    tf1 = TT_seg * 2    # final time for the first transition

    ti2 = TT_seg * 3    # initial time for the second transition
    tf2 = TT_seg * 4    # final time for the second transition

    ti3 = TT_seg * 5    # initial time for the third transition
    tf3 = TT_seg * 6    # final time for the third transition

    xx_ref = np.zeros((ns, TT))    
    uu_ref = np.zeros((ni, TT))

    # Flat part 1 (start at the initial state)
    xx_ref[:, :TT_seg] = state_0
    uu_ref[:, :TT_seg] = input_0

    # Flat part 2 (target state) 
    xx_ref[:, tf1: ti2] = state_1
    uu_ref[:, tf1: ti2] = input_1

    # Flat part 3 (end at the initial state)
    xx_ref[:, tf2: ti3] = state_2
    uu_ref[:, tf2: ti3] = input_2

    # Flat part 4 (end at the initial state)
    xx_ref[:, ti3:] = state_3
    uu_ref[:, ti3:] = input_3

    # Transition parts (smooth trajectory)
    t_transition = np.linspace(0, 1, TT_seg)    # normalized time vector
    smooth_curve = 0.5 * (1 - np.cos(np.pi * t_transition)) # scaled cosine curve for the fisrt transition

    for i in range(4):  
        # First transition
        xx_ref[i, ti1: tf1] = (state_0[i] + smooth_curve * (state_1[i] - state_0[i]))
        
        # Second transition
        xx_ref[i, ti2: tf2] = (state_1[i] + smooth_curve * (state_2[i] - state_1[i]))
        
        # Third transition
        xx_ref[i, ti3: tf3] = (state_2[i] + smooth_curve * (state_3[i] - state_2[i]))

    # Consider a quasi-static trajectory 
    xx_ref[4:8, :] = 0

    # Update control inputs based on dynamics
    for i in range(TT_seg):
        # First transition
        j = ti1 + i
        uu_ref[0, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 1)
        uu_ref[1, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 3)

        # Second transition
        j = ti2 + i
        uu_ref[0, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 1)
        uu_ref[1, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 3)

        # Second transition
        j = ti3 + i
        uu_ref[0, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 1)
        uu_ref[1, j] = alpha * dyn.coupling_sum(xx_ref[:, j], 3)
    

    # PLOT

    fig, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)  # adjust figsize for better vertical space
    fig.canvas.manager.set_window_title('Reference')

    tt_hor = np.linspace(0,tf,TT)

    # Plotting each axis
    labels = ['$z_1$', '$z_2$', '$z_3$', '$z_4$', r'$\dot{z}_1$', r'$\dot{z}_2$', r'$\dot{z}_3$', r'$\dot{z}_4$', '$u_1$', '$u_2$']

    for i, ax in enumerate(axs):
        if i < 8:
            ax.plot(tt_hor, xx_ref[i, :], 'g--', linewidth=2)
        else:
            ax.plot(tt_hor, uu_ref[i - 8, :], 'r--', linewidth=2)
        ax.grid()
        ax.set_xlim([0, tf])
        ax.set_ylabel(labels[i], fontsize=15)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)    # reduce tick label size
    axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

    fig.subplots_adjust(hspace=0.08)    # adjust vertical spacing
    plt.tight_layout()

    plt.show()

    return xx_ref, uu_ref
