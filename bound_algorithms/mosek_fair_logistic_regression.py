# Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
#
# File:      logistic.py
#
# Purpose: Implements logistic regression with regulatization.
#
#          Demonstrates using the exponential cone and log-sum-exp in Optimizer API.
#
#          Plots an example for 2D datasets.
from mosek import *
import numpy as np 
import sys, itertools

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

# Model logistic regression (regularized with full 2-norm of theta)
# X - n x d matrix of data points
# y - length n vector classifying training points
# lamb - regularization parameter
def fairLogisticRegression(X, y, Q, lamb=0.0):
    n, d = int(X.shape[0]), int(X.shape[1])         # num samples, dimension

    with Task() as task:
        # Variables [r; theta; t; u]
        nvar = 1+d+2*n
        task.appendvars(nvar)
        task.putvarboundsliceconst(0, nvar, boundkey.fr, -inf, inf)
        r, theta, t = 0, 1, 1+d
        task.putvarname(r,"r");
        for j in range(d): task.putvarname(theta+j,f"theta[{j}]");
        for j in range(n): task.putvarname(t+j,f"t[{j}]");

        # Add quadratic fairness regularizer: theta^T Q theta
        for i in range(d):
            for j in range(d):
                qval = Q[i, j]
                #if abs(qval) > 1e-10:  # skip zero entries
                task.putqobj(theta + i, theta + j, qval)


        # Objective lambda*r + sum(t)
        task.putobjsense(objsense.minimize)
        task.putcj(r, lamb)
        task.putclist(range(t, t+n), [1.0]*n)

        # Softplus function constraints
        softplus(task, d, n, theta, t, X, y);

        # Regularization
        # Append a sequence of linear expressions (r, theta) to F
        numafe = task.getnumafe()
        task.appendafes(1+d)
        task.putafefentry(numafe, r, 1.0)
        for i in range(d):
          task.putafefentry(numafe + i + 1, theta + i, 1.0)

        # Add the constraint
        task.appendaccseq(task.appendquadraticconedomain(1+d), numafe, None)

        # Solution
        task.writedata('logistic.ptf')
        task.optimize()
        xx = task.getxxslice(soltype.itr, theta, theta+d)

        obj_val = task.getprimalobj(soltype.itr)

        return xx, obj_val