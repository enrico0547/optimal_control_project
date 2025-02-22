import numpy as np
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

QQt = (1)* np.diag([1e3, 1e3, 1e3, 1e3, 1e0, 1e0, 1e0, 1e0])
RRt = (1e-4) * np.eye(ni)
QQT = QQt # we can also choose QQT equal to solution of the DARE (see main)

def stagecost(xx, uu, xx_ref, uu_ref):
  """""
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args:
      - xx: State at time t
      - xx_ref: State reference at time t
      - uu: Input at time t
      - uu_ref: Input reference at time t

    Return: 
      - ll: Cost at xx,uu
      - lx: Gradient of l wrt x, at xx,uu
      - lu: Gradient of l wrt u, at xx,uu
      - QQt, RRt, zero matrix: Matrices Q, R and S at time t (constant)
  """""
  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  return ll.squeeze(), lx.squeeze(), lu.squeeze(), QQt, RRt, np.zeros((ni,ns))

def termcost(xx, xx_ref):
  """""
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args:
      - xx: State at time t
      - xx_ref: State reference at time t

    Return: 
      - llT: Cost at xx,uu
      - lTx: Gradient of l wrt x, at xx,uu
      - QQT: Terminal state weights
  """""
  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

  lTx = QQT@(xx - xx_ref)

  return llT.squeeze(), lTx.squeeze(), QQT


def is_semi_positive_definite(matrix):
    # Calcola gli autovalori della matrice
    eigenvalues = np.linalg.eigvals(matrix)

    # Verifica se tutti gli autovalori sono non negativi
    return np.all(eigenvalues >= 0)