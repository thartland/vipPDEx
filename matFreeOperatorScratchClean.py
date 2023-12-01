# this notebook is to illustrate how to use
# a preconditioner defined as a shell matrix 
# to solve a linear system in a Krylov-subspace method
# the idea is that we need to learn how to define such
# objects as the Hessian is one such shell object
# and we want to solve Hessian linear systems by Krylov-subspace methods

# Import FEniCSx
import dolfinx as dl
import ufl

from mpi4py import MPI
from petsc4py import PETSc

# Import the package of mathematical functions
import numpy as np

# Enable plotting inside the notebook
import matplotlib.pyplot as plt


# Jacobi Operator
class JacobiOp:
    def __init__(self, M):
        self.M = M
        self.Avec = M.createVecLeft()
        self.m, self.n = self.M.getSize()
        # to do: throw error if m not equal to n
        # expected need: grab M.owner_range --> transform into a range() object and pass for parallel runs
        self.Avec.setValues(range(self.m), 1.0 / (M.getDiagonal()))
    def create(self, A):
        mat_size = A.getSize()
    def mult(self, A, x, y):
        y.setValues(range(self.m), x[:] * self.Avec[:])

    #def apply(self, x, y):
    #    y.setValues(range(self.m), x[:] * self.Avec[:])
        
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.mult(A, x, y)

    def getDiagonal(self):
        "D[i] <- A[i,i]"
        return self.Avec[:]




# ---- setup a mass matrix over linear continuous elements over a discretized
# ---- unit square
comm = MPI.COMM_WORLD
nx = 32
ny = nx
mesh = dl.mesh.create_unit_square(comm, nx, ny)
Vh = dl.fem.FunctionSpace(mesh, ('CG', 1))

u_trial = ufl.TrialFunction(Vh)
u_test  = ufl.TestFunction(Vh)

Mvarf = u_trial * u_test * ufl.dx
M = dl.fem.petsc.assemble_matrix(dl.fem.form(Mvarf))
M.assemble()

m, n = M.getSize()

# setup (Jacobi) shell preconditioner
P = PETSc.Mat().create()
P.setSizes([m, n])
P.setType('python')
JacobiShell = JacobiOp(M)
P.setPythonContext(JacobiShell)
P.setUp()

# ---- setup the ksp solver
ksp = PETSc.KSP().create()
ksp.setOperators(M)
ksp.setTolerances(rtol=1.e-14, atol=1.e-14)
ksp.setType('gmres')
ksp.setConvergenceHistory()

# ---- setup the preconditioner
pc = PETSc.PC().create()
pc.setOperators(P)

# ---- set the ksp preconditioner
ksp.setPC(pc)

# ---- setup left- and right-hand side vectors
b    = M.createVecLeft()
b.setValues(range(m), np.random.randn(m))
x = M.createVecRight()

# ---- solve the linear system
ksp.solve(b, x)

# ---- plot linear solve history
residuals = ksp.getConvergenceHistory()
plt.semilogy(residuals)
plt.show()
