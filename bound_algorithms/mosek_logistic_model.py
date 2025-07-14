from mosek import *
import numpy as np 
import sys, itertools
import time
from typing import List, Tuple


class MosekLogisticModel:
    def __init__(self, X: np.ndarray, y: np.ndarray, lamb: float = 0.0):
        self.X_full = X
        self.y = y
        self.n, self.d_full = X.shape
        self.lamb = lamb

        self.r_idx = 0
        self.theta_idx = 1
        self.t_idx = 1 + self.d_full

        self.task = Task()
        self._build_problem()

    def _build_problem(self):
        n, d = self.n, self.d_full
        t_start = self.t_idx
        nvar = 1 + d + 2*n

        self.task.appendvars(nvar)
        self.task.putvarboundsliceconst(0, nvar, boundkey.fr, -inf, inf)

        self.task.putvarname(self.r_idx, "r")
        for j in range(d):
            self.task.putvarname(self.theta_idx + j, f"theta[{j}]")
        for i in range(n):
            self.task.putvarname(t_start + i, f"t[{i}]")

        self.task.putobjsense(objsense.minimize)
        self.task.putcj(self.r_idx, self.lamb)
        self.task.putclist(range(t_start, t_start + n), [1.0]*n)

        softplus(self.task, d, n, self.theta_idx, t_start, self.X_full, self.y)

        # Add regularization cone for r and theta
        numafe = self.task.getnumafe()
        self.task.appendafes(1 + d)
        self.task.putafefentry(numafe, self.r_idx, 1.0)
        for i in range(d):
            self.task.putafefentry(numafe + i + 1, self.theta_idx + i, 1.0)
        self.task.appendaccseq(self.task.appendquadraticconedomain(1 + d), numafe, None)

    def solve(self, fixed_out: List[int], prev_coef = None) -> Tuple[np.ndarray, float]:
        # Fix theta_j = 0 for j in fixed_out
        theta_vec = []
        for j in range(self.d_full):
            idx = self.theta_idx + j
            if j in fixed_out:
                self.task.putvarbound(idx, boundkey.fx, 0.0, 0.0)
                if prev_coef:
                    theta_vec.append(0.0)
            else:
                self.task.putvarbound(idx, boundkey.fr, -inf, inf)
                if prev_coef:
                    theta_vec.append(prev_coef.get(j, 0.0))
        
        if prev_coef is not None:
            self.task.putxxslice(soltype.itr, self.theta_idx, self.theta_idx + self.d_full, theta_vec)

        self.task.optimize()
        coefs = np.array(self.task.getxxslice(soltype.itr, self.theta_idx, self.theta_idx + self.d_full))
        obj = self.task.getprimalobj(soltype.itr)
        return coefs, obj


inf = 0.0

# Adds ACCs for t_i >= log ( 1 + exp((1-2*y[i]) * theta' * X[i]) )
# Adds auxiliary variables, AFE rows and constraints
def softplus(task, d, n, theta, t, X, y):
    nvar = task.getnumvar()
    ncon = task.getnumcon()
    nafe = task.getnumafe()
    task.appendvars(2*n)    # z1, z2
    task.appendcons(n)      # z1 + z2 = 1
    task.appendafes(4*n)    #theta * X[i] - t[i], -t[i], z1[i], z2[i]
    z1, z2 = nvar, nvar+n
    zcon = ncon
    thetaafe, tafe, z1afe, z2afe = nafe, nafe+n, nafe+2*n, nafe+3*n
    for i in range(n):
        task.putvarname(z1+i,f"z1[{i}]")
        task.putvarname(z2+i,f"z2[{i}]")

    # z1 + z2 = 1
    task.putaijlist(range(zcon, zcon+n), range(z1, z1+n), [1]*n)
    task.putaijlist(range(zcon, zcon+n), range(z2, z2+n), [1]*n)
    task.putconboundsliceconst(zcon, zcon+n, boundkey.fx, 1, 1)
    task.putvarboundsliceconst(nvar, nvar+2*n, boundkey.fr, -inf, inf)

    # Affine conic expressions
    afeidx, varidx, fval = [], [], []

    ## Thetas
    for i in range(n):
      for j in range(d):
        afeidx.append(thetaafe + i)
        varidx.append(theta + j)
        fval.append(-X[i][j] if y[i]==1 else X[i][j])

    # -t[i]
    afeidx.extend([thetaafe + i for i in range(n)] + [tafe + i for i in range(n)])
    varidx.extend([t + i for i in range(n)] + [t + i for i in range(n)])
    fval.extend([-1.0]*(2*n))

    # z1, z2
    afeidx.extend([z1afe + i for i in range(n)] + [z2afe + i for i in range(n)])
    varidx.extend([z1 + i for i in range(n)] + [z2 + i for i in range(n)])
    fval.extend([1.0]*(2*n))

    # Add the expressions
    task.putafefentrylist(afeidx, varidx, fval)

    # Add a single row with the constant expression "1.0"
    oneafe = task.getnumafe()
    task.appendafes(1)
    task.putafeg(oneafe, 1.0)

    # Add an exponential cone domain
    expdomain = task.appendprimalexpconedomain()

    # Conic constraints
    acci = task.getnumacc()
    for i in range(n):
      task.appendacc(expdomain, [z1afe+i, oneafe, thetaafe+i], None)
      task.appendacc(expdomain, [z2afe+i, oneafe, tafe+i], None)
      task.putaccname(acci,  f"z1:theta[{i}]")
      task.putaccname(acci+1,f"z2:t[{i}]")
      acci += 2

      