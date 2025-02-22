import numpy as np
import cvxpy as cp 
import dynamics as dyn

ni = dyn.ni
ns = dyn.ns

def ltv_LQR(AA, BB, QQ, RR, SS, QQf, qqf, qq, cc, rr, TT):
    """""
    LQR solver - linear time-varying systems with quadratic cost functions
    
    Solves the discrete-time finite-horizon linear quadratic regulator (LQR) problem
    for a system described by time-varying dynamics and cost matrices
    The algorithm performs backward recursion to compute the optimal state-feedback gain K and feedforward term sigma

    Args:
        - AA: Linearized dynamics matrix.
        - BB: Linearized input matrix.
        - QQ: State weights.
        - RR: Input weights.
        - SS: Cross-term weights. (Zero matrix)
        - QQf: Terminal state weights
        - qqf: Terminal linear state penalty.
        - qq: Linear state penalty.
        - cc: State linearization bias.
        - rr: Linear input penalty
        - TT: Time horizon.

    Returns:
        - KK: Optimal feedback gain matrices.
        - sigma: Feedforward terms.
    """""
    PP = np.zeros((ns, ns, TT))
    KK = np.zeros((ni, ns, TT))
    pp = np.zeros((ns, 1, TT))
    sigma = np.zeros((ni, 1, TT))
    
    PP[:, :, -1] = QQf
    pp[:, :, -1] = qqf
 
    for tt in reversed(range(TT - 1)): 
        QQt = QQ[:, :, tt]
        RRt = RR[:, :, tt]
        AAt = AA[:, :, tt]
        BBt = BB[:, :, tt]
        SSt = SS[:, :, tt]  # always equal to 0, see the cost function definition of lux
        PPtp = PP[:, :, tt + 1]
        qqt = qq[:, tt] # qq is (8, 1)
        pptp = pp[:, :, tt + 1]  
        cct = cc[:, :, tt]  # cc is (8, 1)
        rrt = rr[:, tt] # rr is (2, 1)             

        KKt = - np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) @ (SSt + BBt.T @ PPtp @ AAt)
        PPt = AAt.T @ PPtp @ AAt - KKt.T @ (RRt + BBt.T @ PPtp @ BBt) @ KKt + QQt
        sigmat = - np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) @ (rrt.reshape(-1, 1) + BBt.T @ pptp + (BBt.T @ PPtp @ cct).reshape(-1, 1))
        ppt = qqt.reshape(-1, 1) + AAt.T @ pptp + (AAt.T @ PPtp @ cct).reshape(-1, 1) - KKt.T @ (RRt + BBt.T @ PPtp @ BBt) @ sigmat

        sigma[:, :, tt] = sigmat
        pp[:, :, tt] = ppt
        PP[:, :, tt] = PPt
        KK[:, :, tt] = KKt
        
    return KK, sigma

def solver_mpc(AA, BB, QQ, RR, QQf, xxt, u_max, u_min,  T_pred, uu_traj, TT):
    """""
        Linear MPC solver - Constrained LQR

        Given a measured state xxt measured at t
        gives back the optimal input to be applied at t

        Args
          - AA, BB: linear dynamics
          - QQ,RR,QQf: cost matrices
          - xxt: initial condition (at time t)
          - x1_max, x1_min: position constraints
          - x2_max, x2_min: velocity constraints
          - u_max, u_min: input constraints
          - T_pred: time (prediction) horizon
          - uu_traj: reference input trajectory
          - TT: current time index

        Returns
          - u_mpc: input to be applied at t
          - xx_mpc, uu_mpc: predicted trajectory
    """""
    
    xxt = xxt.squeeze()

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []

    for tt in range(T_pred-1):
        ref_index = min(TT + tt, uu_traj.shape[1] - 1)
        ref_u = uu_traj[:, ref_index]  # Use last valid reference if exceeding

        cost += cp.quad_form(xx_mpc[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt], RR)
        constr += [xx_mpc[:,tt+1] == AA@xx_mpc[:,tt] + BB@uu_mpc[:,tt], # dynamics constraint
                (uu_mpc[:,tt] + ref_u) <= u_max,  # other constraints
                (uu_mpc[:,tt] + ref_u) >= u_min
                ]
        
    # Sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)
    constr += [xx_mpc[:,0] == xxt]

    problem = cp.Problem(cp.Minimize(cost), constr) 
    problem.solve()

    if problem.status == "infeasible":
    # Otherwise, problem.value is inf or -inf, respectively.
        print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value