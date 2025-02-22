import numpy as np

import matplotlib.pyplot as plt

import dynamics as dyn

import cost as cst
from cost import QQt, RRt

import armijo

import reference_trajectory as ref_gen

import animation


import control as ctrl

import solver

# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})


#######################################
# Algorithm parameters
#######################################

max_iters = 20
fixed_stepsize = 1e-1

# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 1.0
cc = 0.5
beta = 0.7
armijo_maxiters = 20  # number of Armijo iterations

term_cond = 1e-6

# Settings
visu_descent_plot = True
visu_animation = True

step_reference = False
mpc_on = True  # if True, the MPC is computed and animated

#######################################
# Trajectory parameters
#######################################

tf = 1.0  # final time in seconds

dt = dyn.dt # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf/dt) # discrete-time samples

# Definition of vector useful for the computation of the deltas:
KK = np.zeros((ni, ns, TT, max_iters))
sigma = np.zeros((ni, 1, TT, max_iters))
A = np.zeros((ns, ns, TT, max_iters))
B = np.zeros((ns, ni, TT, max_iters))
cc_t = np.zeros((ns, 1, TT, max_iters)) # TODO 
QQ = np.zeros((ns, ns, TT, max_iters))
RR = np.zeros((ni, ni, TT, max_iters))
SS = np.zeros((ni, ns, TT, max_iters))
qq = np.zeros((ns, TT, max_iters))
rr = np.zeros((ni, TT, max_iters))
Hessian = np.zeros((ns, ns, ns, TT, max_iters))
tensor = np.zeros((ns, ns, TT, max_iters))

# Definition of the reference trajectory
xx_ref = np.zeros((ns, TT))
uu_ref = np.zeros((ni, TT))

# Symbolic linearization of the dynamics 
A_sym , B_sym = dyn.linearize_dynamics_symbolic()
Hessian_state = dyn.compute_hessian_symbolic()

######################################
# Reference curve
######################################

if(step_reference):
  # Given the two set of inputs, generate a step reference between the correspondent equilibrium points
  u_initial = np.array([0.0, 0.0, 0.0, 0.0])
  u_final = np.array([0.0, 35.0, 0.0, -20.0])

  xx_ref, uu_ref = ref_gen.gen(tf, u_initial, u_final)

else:
  # Given the four set of inputs, generate a smooth reference between the correspondent equilibrium points
  u0 = np.array([0.0, -50.9, 0.0, -30.0])
  u1 = np.array([0.0, 100.0, 0.0, 45.6])
  u2 = np.array([0.0, 50.0, 0.0, -20.6])
  u3 = np.array([0.0, -30.0, 0.0, -10.6])

  xx_ref, uu_ref = ref_gen.gen_swing(tf, u0, u1, u2, u3)

#############################################
#Compute QQT as solution of the DARE
##############################################

AA_ref, BB_ref = dyn.linearize_dynamics_numeric(xx_ref[:,TT-1], uu_ref[:,TT-1], A_sym, B_sym)

QQT = ctrl.dare(AA_ref,BB_ref, QQt, RRt)[0] # solvers uses QQT instead of Q_T

x0 = xx_ref[:,0]

######################################
# Initial guess
######################################

xx_init = np.repeat(xx_ref[:,0].reshape(-1,1), TT, axis=1)
uu_init = np.repeat(uu_ref[:,0].reshape(-1,1), TT, axis=1)

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))
uu = np.zeros((ni, TT, max_iters))

lmbd = np.zeros((ns, TT, max_iters))  # lambdas - costate seq.

deltax = np.zeros((ns, TT, max_iters))  # state corrections
deltau = np.zeros((ni, TT, max_iters))  # input corrections

dJ = np.zeros((ni,TT, max_iters)) # dJ - gradient of J wrt u

JJ = np.zeros(max_iters)  # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction

######################################
# Main
######################################

print('-*-*-*-*-*-')

kk = 0

xx[:,:,0] = xx_init
uu[:,:,0] = uu_init

for kk in range(max_iters-1):
  JJ[kk] = 0
  
  # Calculate cost 
  for tt in range(TT-1):
    temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
    JJ[kk] += temp_cost
    
    _, qq[:, tt, kk], rr[:, tt, kk], QQ[:, :, tt, kk], RR[:, :, tt, kk], SS[:, :, tt, kk] = (
          cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt]))  

    # Jacobians
    A[:, :, tt, kk], B[:, :, tt, kk] = dyn.linearize_dynamics_numeric(xx[:, tt, kk], uu[:, tt, kk], A_sym, B_sym) 
    # Hessian
    Hessian[:, :, :, tt, kk] = dyn.compute_hessian_numeric(xx[:, tt, kk], uu[:, tt, kk], Hessian_state)


  temp_cost = cst.termcost(xx[:,-1,kk], xx_ref[:,-1])[0]
  JJ[kk] += temp_cost

  ##################################
  # Descent direction calculation
  ##################################

  # Solve affine LQR for descent direction
  _, q_T, Q_T = cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])
  q_T = q_T.reshape(-1, 1)

  # Solve the costate equation to compute Q_t, R_t, S_t and dJ
  lmbd[:, TT - 1, kk] = q_T.squeeze().copy()

  for tt in reversed(range(TT-1)):
    A_t = A[:, :, tt, kk]
    B_t = B[:, :, tt, kk]
    lmbd_p = lmbd[:, tt+1, kk].reshape(-1, 1)
    q_t = qq[:, tt, kk].reshape(-1, 1)
    r_t = rr[:, tt, kk].reshape(-1, 1)
    Q_t = QQ[:, :, tt, kk]

    lmbd_temp = A_t.T @ lmbd_p + q_t # costate equation
    dJ_temp = B_t.T @ lmbd_p + r_t # gradient of J wrt u

    for ll in range(ns):
      tensor[:, :, tt, kk] += lmbd_temp[ll, 0] * Hessian[ll, :, :, tt, kk]
    Q_temp = Q_t + tensor[:, :, tt, kk]

    # Reguralized method if the matrix is not positive definite
    if cst.is_semi_positive_definite(Q_temp):
      QQ[:, :, tt, kk] = Q_temp
      # print("Positive definite")
    else:
      QQ[:, :, tt, kk] = Q_t
      # print("NOT positive definite")

    dJ[:, tt, kk] = dJ_temp.squeeze()
    lmbd[:, tt, kk] = lmbd_temp.squeeze()

  KK[:, :, :, kk], sigma[:, :, :, kk] = solver.ltv_LQR(A[:, :, :, kk], B[:, :, :, kk], QQ[:, :, :, kk], RR[:, :, :, kk], SS[:, :, :, kk], Q_T,  # QQT is the solution of the DARE
                                                          q_T, qq[:, :, kk], cc_t[:, :, :, kk], rr[:, :, kk], TT)

  ####################################
  ## Forward pass  (Compute deltas) ##
  ####################################

  for tt in range(TT - 1):
    A_t = A[:, :, tt, kk]
    B_t = B[:, :, tt, kk]
    K_t = KK[:, :, tt, kk]
    sigma_t = sigma[:, :, tt, kk]
    deltax_t = deltax[:, tt, kk].reshape(-1, 1)
    
    # Compute input correction
    deltau_temp = (K_t @ deltax_t + sigma_t).squeeze()

    # Compute state correction
    deltax_temp = (A_t @ deltax_t + B_t @ deltau_temp.reshape(-1, 1)).squeeze()

    # Compute descent direction
    descent[kk] += deltau_temp.T@deltau_temp
    descent_arm[kk] += dJ[:,tt,kk].T@ deltau_temp

    # Update the state and input sequences
    deltax[:, tt+1, kk] = deltax_temp.squeeze()
    deltau[:, tt, kk] = deltau_temp.squeeze()


  ##################################
  # Stepsize selection - ARMIJO
  ##################################
  
  deltau[:,-1,kk] = deltau[:,-2,kk]

  if Armijo:

    stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                xx_ref, uu_ref, x0, 
                                xx[:,:,kk], uu[:,:,kk], JJ[kk], descent_arm[kk], 
                                KK[:, :, :, kk], sigma[:, :, :, kk], deltau[:, :, kk], visu_descent_plot)
  else:
     stepsize = fixed_stepsize

  
  ############################
  # Update the current solution
  ############################

  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))
  xx_temp[:,0] = x0 # initial condition

  # Update the state and input sequences using feedback integration
  for tt in range(TT-1):
      uu_temp[:,tt] = uu[:,tt,kk] + KK[:, :, tt, kk] @(xx_temp[:,tt] - xx[:,tt,kk]) + (stepsize * sigma[:, :, tt, kk]).reshape(2, )
      # uu_temp[:,tt] = uu[:, tt, kk] + stepsize * deltau[:, tt, kk]
      xx_temp[:,tt+1]  = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt]).squeeze()
      
  # Copy the penultimate element of the input sequence to the last element
  uu_temp[:,-1] = uu_temp[:,-2] 

  # Update the state and input sequences
  xx[:,:,kk+1] = xx_temp.copy()
  uu[:,:,kk+1] = uu_temp.copy()


  ############################
  # Termination condition
  ############################

  print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk
    print("Termination condition reached at iteraration", max_iters) 

    break

if kk == 0:
  max_iters = 1

############################
# TASK 3
############################

xx_traj = np.zeros((ns, TT))
xx_traj = xx[:,:, max_iters]
uu_traj = np.zeros((ni, TT))
uu_traj = uu[:,:, max_iters]  # inizialize at the last iteration

xx_track = np.zeros((ns, TT))
uu_track = np.zeros((ni, TT))

A_traj = np.zeros((ns, ns, TT))
B_traj = np.zeros((ns, ni, TT))

QQt_reg = (1)* np.diag([1e3, 1e3, 1e3, 1e3, 1e0, 1e0, 1e0, 1e0]) 
RRt_reg = (1e-4) * np.eye(ni)

PPt = np.zeros((ns, ns, TT))
PPtP = np.zeros((ns, ns))
KK_reg = np.zeros((ni, ns, TT))

measure_noise = np.random.normal(0, 0.0001, (ns, TT))

# Step 1: linearize the dynamics around the optimal trajectory
for tt in range(TT) :
  A_traj[:, :, tt], B_traj[:, :, tt] = dyn.linearize_dynamics_numeric(xx_traj[:, tt], uu_traj[:, tt], A_sym, B_sym)

# Step 2: LQ backward pass
QQT_reg = ctrl.dare(A_traj[:, :, TT-1], B_traj[:, :, TT-1], QQt_reg, RRt_reg)[0]
PPt[:, :, TT - 1] = QQT_reg

for tt in reversed(range(TT - 1)):
  PPtP = PPt[:, :, tt + 1]
  PPt[:, :, tt] = (QQt_reg + A_traj[:, :, tt].T @ PPtP @ A_traj[:, :, tt] 
                   - (A_traj[:, :, tt].T @ PPtP @ B_traj[:, :, tt]) @ (np.linalg.inv(RRt_reg + B_traj[:, :, tt].T @ PPtP @ B_traj[:, :, tt] )) @ (B_traj[:, :, tt].T @ PPtP @ A_traj[:, :, tt])) 

for tt in range(TT - 1) :
  PPtP = PPt[:, :, tt + 1]
  KK_reg[:, :, tt] = - np.linalg.inv(RRt_reg +B_traj[:, :, tt].T @ PPtP @ B_traj[:, :, tt]) @ (B_traj[:, :, tt].T @ PPtP @ A_traj[:, :, tt])

# Step 3: track the generated optimal trajectory
# inizialize the system far from the equilibrium point
x0 = xx_traj[:, 0] + 7 * xx_traj[:, -1]
# x0 = - xx_traj[:, 0] + 6 * xx_traj[:, -1]

xx_track[:, 0] = x0
for tt in range(TT - 1) :
  uu_track[:, tt] = uu_traj[:, tt] + KK_reg[:, :, tt] @ (xx_track[:, tt] - xx_traj[:, tt] + measure_noise[:, tt])
  xx_track[:, tt + 1] = dyn.dynamics(xx_track[:, tt], uu_track[:, tt]).squeeze()


############################
# TASK 4: MPC
############################

# Parameters initialization
Tsim = TT # simulation horizon

T_pred = 10 # MPC Prediction horizon
umax = 1.1 * max(max(uu_traj[0, :], key=abs), max(uu_traj[1, :], key=abs))  # Upper input constraint
umin = -umax  # Lower input constraint

# Cost
QQ_mpc = (1)* np.diag([1e3, 1e3, 1e3, 1e3, 1e0, 1e0, 1e0, 1e0]) 
RR_mpc = (1e-4) * np.eye(ni) 
QQT_mpc = ctrl.dare(A_traj[:, :, TT-1], B_traj[:, :, TT-1], QQ_mpc, RR_mpc)[0]

# MPC
xx_real_mpc = np.zeros((ns,Tsim))
uu_real_mpc = np.zeros((ni,Tsim))

xx_mpc = np.zeros((ns, T_pred, Tsim))

xx_real_mpc[:,0] = x0

if mpc_on :
  for tt in range(Tsim-1):
    # System evolution - real with MPC
    xx_t_mpc = xx_real_mpc[:,tt] - xx_traj[:,tt]  # get initial condition, global change of coordinates

    # Solve MPC problem - apply first input
    if tt%10 == 0:  # print every 5 time instants
      print('MPC:\t t = {}'.format(tt))

    uu_t_mpc, xx_mpc[:,:,tt] = solver.solver_mpc(A_traj[:, :, tt], B_traj[:, :, tt], QQ_mpc, RR_mpc, QQT_mpc, xx_t_mpc, umax, umin, T_pred, uu_traj, tt)[:2]
    
    uu_real_mpc[:,tt] = uu_traj[:,tt] + uu_t_mpc  # global change of coordinates
    
    xx_real_mpc[:,tt+1] = dyn.dynamics(xx_real_mpc[:,tt], uu_real_mpc[:,tt]).squeeze()


############################
# Plots
############################

# Cost
plt.figure('descent direction')
plt.plot(np.arange(max_iters+1), descent[:max_iters+1])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# Descent
plt.figure('cost')
plt.plot(np.arange(max_iters+1), JJ[:max_iters+1])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# Optimal trajectory
fig1, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)  # adjust figsize for better vertical space
fig1.canvas.manager.set_window_title('Optimal trajectory vs Desired Curve')

xx_star = xx[:,:,max_iters]
uu_star = uu[:,:,max_iters]
uu_star[:, -1] = uu_star[:, -2] # for plotting purposes

tt_hor = np.linspace(0,tf,TT)

# Plotting each axis
labels = ['$z_1$', '$z_2$', '$z_3$', '$z_4$', r'$\dot{z}_1$', r'$\dot{z}_2$', r'$\dot{z}_3$', r'$\dot{z}_4$', '$u_1$', '$u_2$']
for i, ax in enumerate(axs):
    if i < 8:
        ax.plot(tt_hor, xx_star[i, :], linewidth=2)
        ax.plot(tt_hor, xx_ref[i, :], 'g--', linewidth=2)
    else:
        ax.plot(tt_hor, uu_star[i - 8, :], 'r', linewidth=2)
        ax.plot(tt_hor, uu_ref[i - 8, :], 'r--', linewidth=2)
    ax.grid()
    ax.set_xlim([0, tf])
    ax.set_ylabel(labels[i], fontsize=15)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)  # reduce tick label size
axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

fig1.subplots_adjust(hspace=0.08)  # adjust vertical spacing
plt.tight_layout()

plt.show(block=False)

# Intermediate trajectories
fig, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)
fig.canvas.manager.set_window_title('Intermediate trajectory - Iter=1 vs Iter=2')

for i, ax in enumerate(axs):
    if i < 8:
        ax.plot(tt_hor, xx[i, :, 2], 'r', linewidth=2)
        ax.plot(tt_hor, xx[i, :, 1], 'b--', linewidth=1)
        ax.plot(tt_hor, xx_ref[i, :], 'g--', linewidth=1)
    else:
        ax.plot(tt_hor, uu[i - 8, :, 2], 'r', linewidth=2)
        ax.plot(tt_hor, uu[i - 8, :, 1], 'b--', linewidth=1)
        ax.plot(tt_hor, uu_ref[i - 8, :], 'g--', linewidth=1)
    ax.grid()
    ax.set_xlim([0, tf])
    ax.set_ylabel(labels[i], fontsize=15)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

fig.subplots_adjust(hspace=0.08)
plt.tight_layout()

plt.show(block=False)

# Track trajectory
fig2, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)
fig2.canvas.manager.set_window_title('LQR solution - task3')

xx_star = xx_track[:, :]
uu_star = uu_track[:, :]
uu_star[:, -1] = uu_star[:, -2]

for i, ax in enumerate(axs):
    if i < 8:
        ax.plot(tt_hor, xx_star[i, :], linewidth=2)
        ax.plot(tt_hor, xx_traj[i, :], 'g--', linewidth=2)
    else:
        ax.plot(tt_hor, uu_star[i - 8, :], 'r', linewidth=2)
        ax.plot(tt_hor, uu_traj[i - 8, :], 'r--', linewidth=2)
    ax.grid()
    ax.set_xlim([0, tf])
    ax.set_ylabel(labels[i], fontsize=15)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

fig2.subplots_adjust(hspace=0.08)
plt.tight_layout()

plt.show(block=False)

# Tracking error
fig5, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)
fig5.canvas.manager.set_window_title('LQR - Tracking Error')

xx_err = xx_track[:, :] - xx_traj[:, :]
uu_err = uu_track[:, :] - uu_traj[:, :]
uu_err[:, -1] = uu_err[:, -2]

for i, ax in enumerate(axs):
  if i < 8:
    ax.plot(tt_hor, xx_err[i, :], linewidth=1.5)
  else:
    ax.plot(tt_hor, uu_err[i-8, :], linewidth=1.5)
  ax.set_xlim([0, tf])
  ax.grid()
  ax.set_ylabel(labels[i], fontsize=15)
  ax.yaxis.set_label_coords(-0.05, 0.5)
  ax.tick_params(axis='both', which='major', labelsize=10)
axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

fig5.subplots_adjust(hspace=0.08)
plt.tight_layout()

plt.show()

if mpc_on:
# Plots MPC
  fig3, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)
  fig3.canvas.manager.set_window_title('MPC solution - task4')

  time = np.arange(Tsim) * 1e-4
  xx_star = xx_real_mpc[:, :]
  uu_star = uu_real_mpc[:, :]
  uu_star[:, -1] = uu_star[:, -2]

  for i, ax in enumerate(axs):
      if i < 8:
          ax.plot(time, xx_star[i, :], linewidth=2)
          ax.plot(time, xx_traj[i, :], 'g--', linewidth=2)
      else:
          ax.plot(time, uu_star[i-8, :], linewidth=2)
          ax.plot(time, uu_traj[i-8, :], 'g--', linewidth=2)
          ax.plot(time, np.ones(Tsim)*umax, 'g--', linewidth=1.5 )
          ax.plot(time, np.ones(Tsim)*umin, 'g--', linewidth=1.5 )
      ax.set_xlim([0, Tsim*1e-4])
      ax.grid()
      ax.set_ylabel(labels[i], fontsize=15)
      ax.yaxis.set_label_coords(-0.05, 0.5)
      ax.tick_params(axis='both', which='major', labelsize=10)
  axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

  fig3.subplots_adjust(hspace=0.08)
  plt.tight_layout()

  plt.show(block=False)

# Tracking error
  fig4, axs = plt.subplots(ns + ni, 1, figsize=(10, 16), sharex=True)
  fig4.canvas.manager.set_window_title('MPC - Tracking Error')

  xx_err = xx_real_mpc[:, :] - xx_traj[:, :]
  uu_err = uu_real_mpc[:, :] - uu_traj[:, :]
  uu_err[:, -1] = uu_err[:, -2]

  for i, ax in enumerate(axs):
      if i < 8:
          ax.plot(time, xx_err[i, :], linewidth=1.5)
      else:
          ax.plot(time, uu_err[i-8, :], linewidth=1.5)
      ax.set_xlim([0, Tsim*1e-4])
      ax.grid()
      ax.set_ylabel(labels[i], fontsize=15)
      ax.yaxis.set_label_coords(-0.05, 0.5)
      ax.tick_params(axis='both', which='major', labelsize=10)
  axs[-1].set_xlabel('time', fontsize=15, labelpad=0.05)

  fig3.subplots_adjust(hspace=0.08)
  plt.tight_layout()

  plt.show()

if visu_animation:
  animation.animate_dynamic_system(xx_star[:4, :])