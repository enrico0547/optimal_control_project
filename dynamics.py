import numpy as np
import sympy as sp

# Model Parameters
ns = 8
ni = 2

dt = 1e-4   # discretization stepsize

# Dynamics parameters
m = 0.1
m_act = 0.2
d = 0.2
alpha = 128 * 0.25
c = 0.1

def coupling_sum(xx, i):
    """""
    Computes coupling term between reference points
    for the i-th acceleration component
    Args:
        - xx: State vector
        - i: Reference to the i-th point
    
    Returns:
        - sum: Coupling term
    """""
    sum = 0

    L = np.abs((-1) - i) * d
    sum += xx[i] / ( L * (L**2 - (xx[i])**2) )

    for j in range(4):
        if j != i:
            L = np.abs(j - i) * d
            sum += (xx[i] - xx[j]) / ( L * (L**2 - (xx[i] - xx[j])**2) )
    
    L = np.abs(4 - i) * d
    sum += xx[i] / ( L * (L**2 - (xx[i])**2) )

    return sum

def dynamics(xx, uu, dt=dt):
    """""
    Computes the dynamics of the flexible surface
    Args:
        - xx: State vector
        - uu: Input vector

    Returns:
        - xxp: Discretized state vector
    """""
    xxp = np.zeros((ns, 1))

    # Discretization of the dynamics - Forward Euler
    # x(k+1) = x(k) + dt * x_dot(k)
    xxp[0] = xx[0] + dt * xx[4]
    xxp[1] = xx[1] + dt * xx[5]
    xxp[2] = xx[2] + dt * xx[6]
    xxp[3] = xx[3] + dt * xx[7]

    xxp[4] = xx[4] + dt * (- alpha * coupling_sum(xx, 0) - c * xx[4]) / m
    xxp[5] = xx[5] + dt * (uu[0] - alpha * coupling_sum(xx, 1) - c * xx[5]) / m_act
    xxp[6] = xx[6] + dt * (- alpha * coupling_sum(xx, 2) - c * xx[6]) / m
    xxp[7] = xx[7] + dt * (uu[1] - alpha * coupling_sum(xx, 3) - c * xx[7]) / m_act

    return xxp


def linearize_dynamics_symbolic():
    """""
    Linearize the dynamics of the double pendulum

    Returns:
        - A_func: Function that computes the Jacobian of x_dot w.r.t. x
        - B_func: Function that computes the Jacobian of x_dot w.r.t. u
    """""
    # Define symbolic variables
    z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2 = sp.symbols('z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2')

    x = sp.Matrix([z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot]) # state variables
    u = sp.Matrix([u1, u2]) # input variables
    
    # Continuous dynamics
    x_dot_sym = sp.Matrix([z1_dot, z2_dot, z3_dot, z4_dot, 0, 0, 0, 0])
    
    x_dot_sym[4] = (- alpha * coupling_sum(x, 0) - c * z1_dot) / m
    x_dot_sym[5] = (u1 - alpha * coupling_sum(x, 1) - c * z2_dot) / m_act
    x_dot_sym[6] = (- alpha * coupling_sum(x, 2) - c * z3_dot) / m
    x_dot_sym[7] = (u2 - alpha * coupling_sum(x, 3) - c * z4_dot) / m_act
    
    A_sym = x_dot_sym.jacobian(x)   # Jacobian of x_dot w.r.t. x
    B_sym = x_dot_sym.jacobian(u)   # Jacobian of x_dot w.r.t. u
    
    A_func = sp.lambdify((z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2 ), A_sym, 'numpy')    # Convert A_sym to a function
    B_func = sp.lambdify((z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2 ), B_sym, 'numpy')    # Convert B_sym to a function
    
    return A_func, B_func

def linearize_dynamics_numeric(xx, uu, A_func, B_func, dt=dt):
    """""
    Linearize the dynamics of the double pendulum
    Args:
        - xx: State vector
        - uu: Input vector
        - A_func: Function that computes the Jacobian of x_dot w.r.t. x
        - B_func: Function that computes the Jacobian of x_dot w.r.t. u
        - dt: Discretization time

    Returns:
        - A_dis: Discretized Jacobian of x_dot w.r.t. x
        - B_dis: Discretized Jacobian of x_dot w.r.t. u
    """""
    # Define symbolic variables
    z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot = xx # state variables
    u1, u2 = uu # input variables

    A = A_func(z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2 )    # Jacobian of x_dot w.r.t. x
    B = B_func(z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot,  u1, u2 )    # Jacobian of x_dot w.r.t. u

    # Discretize the dynamics
    A_numeric = np.eye(ns) + A*dt 
    B_numeric = B*dt 
    
    return A_numeric, B_numeric

def compute_hessian_symbolic():
    """""
    Compute the symbolic Hessian tensor of the dynamics with respect to the state variables.

    Returns:
        - H_func: Function that computes the Hessian tensor for a given state and input.
    """""
    # Define symbolic variables
    z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot, u1, u2 = sp.symbols(
        'z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot, u1, u2')

    x = sp.Matrix([z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot])  # state variables
    u = sp.Matrix([u1, u2])  # input variables

    # Continuous dynamics
    x_dot_sym = sp.Matrix([z1_dot, z2_dot, z3_dot, z4_dot, 0, 0, 0, 0])

    x_dot_sym[4] = (- alpha * coupling_sum(x, 0) - c * z1_dot) / m
    x_dot_sym[5] = (u1 - alpha * coupling_sum(x, 1) - c * z2_dot) / m_act
    x_dot_sym[6] = (- alpha * coupling_sum(x, 2) - c * z3_dot) / m
    x_dot_sym[7] = (u2 - alpha * coupling_sum(x, 3) - c * z4_dot) / m_act

    # Compute the Hessian of x_dot_sym with respect to x
    H_sym = [[[sp.diff(x_dot_sym[i], x[j], x[k]) for k in range(ns)] for j in range(ns)] for i in range(ns)]

    H_func = sp.lambdify((z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot, u1, u2), H_sym, 'numpy')

    return H_func


def compute_hessian_numeric(xx, uu, H_func):
    """""
    Compute the numeric Hessian tensor for the given state and input.

    Args:
        - xx: State vector
        - uu: Input vector
        - H_func: Function that computes the Hessian tensor symbolically

    Returns:
        - H_numeric: Hessian tensor evaluated at the given state and input
    """""
    z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot = xx  # state variables
    u1, u2 = uu  # input variables

    H_numeric = np.array(H_func(z1, z2, z3, z4, z1_dot, z2_dot, z3_dot, z4_dot, u1, u2))

    return H_numeric