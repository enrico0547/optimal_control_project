import numpy as np
import dynamics as dyn

from dynamics import coupling_sum

x0 = np.array([0.0, 0.0, 0.0, 0.0]) # Initial guess for the state vector

# To find the equilibrium points, we need to find the points where the velocity and the acceleration are zero
d = dyn.d
alpha = dyn.alpha

def grad_sum(xx, i):
    """""
    Computes the derived couping term

    Args:
        - xx: State vector
        - i: Reference to the i-th point
    
    Returns:
        - sum: Derived coupling term
    """""
    sum = 0
    
    L = np.abs((-1) - i) * d
    sum += (L**2 + xx[i]**2) / ( L * (L**2 - xx[i] **2)**2 )

    for j in range(4):
        if j != i:    
            L = np.abs(j - i) * d
            sum += (L**2 + (xx[i] - xx[j])**2) / ( L * (L**2 - (xx[i] - xx[j])**2)**2 )

    L = np.abs(4 - i) * d
    sum += (L**2 + xx[i] **2) / ( L * (L**2 - xx[i]**2)**2 )
    
    return sum

def equilibrium_function(xx, uu):
    """""
    Computes the residual vector for the equilibrium condition

    Args:
        - xx: State vector
        - uu: Input vector

    Returns:
        - rr: Residual vector, representing the deviation from equilibrium
              The equilibrium condition is satisfied when rr = 0
    """""
    rr = np.zeros(4)

    for i in range(4):
        rr[i] = uu[i] - alpha*coupling_sum(xx, i)

    return rr

def equilibrium_jacobian(xx):
    """""
    Computes the Jacobian matrix of the residual function with respect to the state vector
    The Jacobian is the transpose of the gradient, capturing how the residual vector changes with respect to the states

    Args:
        - xx: State vector (shape: [4,]), current state of the system

    Returns:
        - J: Jacobian matrix
    """""
    J = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if i == j:
                J[i,j] = - alpha*grad_sum(xx, i)    # derivative of rr[i] wrt xx[i]
            else:
                L = np.abs(j - i) * d
                J[i,j] = alpha * (L**2 + (xx[i]-xx[j])**2) / (L * (L**2 - (xx[i] - xx[j])**2)**2)   # derivative of rr[i] wrt xx[j]

    return J

def equilibrium_points(uu0):
    """""
    Uses the Newton-Raphson method to find equilibrium points of the system

    Args:
        - uu0: Input vector

    Returns:
        - xx: Equilibrium state vector
    """""
    tol=1e-6
    max_iter=100

    xx = x0
    uu = uu0
    for _ in range(max_iter):
        rr = equilibrium_function(xx, uu)
        J = equilibrium_jacobian(xx)

        dx = - np.linalg.inv(J) @ rr
        xx += dx
        if np.linalg.norm(dx) < tol:
            break

    return xx