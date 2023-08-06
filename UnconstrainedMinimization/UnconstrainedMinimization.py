# Import FEniCSx
import dolfinx as dl
import ufl

from mpi4py import MPI
from petsc4py import PETSc

# Import the package of mathematical functions
import math
import numpy as np

# Enable plotting inside the notebook
import matplotlib.pyplot as plt
import pyvista as pv
pv.set_jupyter_backend('static')


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
iAmRoot = rank == 0
# Define the finite element mesh. The mesh size h is 1/nx
nx = 16
ny = nx
    
mesh = dl.mesh.create_unit_square(comm, nx, ny, dl.mesh.CellType.triangle)
cells, types, coords = dl.plot.create_vtk_mesh(mesh, mesh.topology.dim)
grid = pv.UnstructuredGrid(cells, types, coords)

# Create plotter
window_size=[400,400]
plotter = pv.Plotter(window_size=window_size)
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.add_title('Finite Element Mesh', font='courier', color='w',
                   font_size=10)
plotter.background_color = "black"
plotter.show()
plotter.close()

# Define the finite element space V_h as the space of piecewise linear functions on the elements of the mesh.
degree = 1
Vh = dl.fem.FunctionSpace(mesh, ("CG", degree))
uh  = dl.fem.Function(Vh)

glbSize = uh.vector.size
if iAmRoot:
    print("Number of dofs", glbSize)



# ---- algebraically specify the Dirichlet boundary conditions
facet_dim = mesh.topology.dim-1
facets_D = dl.mesh.locate_entities_boundary(mesh, dim=facet_dim, \
                                        marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), \
                                        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[0], 1.0))),
                                        np.isclose(x[1], 1.0)))

# ---- specify the finite element dofs that the Dirichlet conditions will be applied
dofs_D = dl.fem.locate_dofs_topological(V=Vh, entity_dim=facet_dim, entities=facets_D)

# ---- specify the function values on the Dirichlet boundary
# ---- specify a null-function on the Dirichlet boundary
u_bc  = dl.fem.Constant(mesh, PETSc.ScalarType(0.0))
u_bc0 = dl.fem.Constant(mesh, PETSc.ScalarType(0.0)) # homogeneous
bcs  = [dl.fem.dirichletbc(u_bc, dofs_D, Vh)]
bcs0 = [dl.fem.dirichletbc(u_bc0, dofs_D, Vh)]

# ---- the energy functional J
k1 = dl.fem.Constant(mesh, PETSc.ScalarType(0.05))
k2 = dl.fem.Constant(mesh, PETSc.ScalarType(1.0))
f  = dl.fem.Constant(mesh, PETSc.ScalarType(1.0))

Jform = PETSc.ScalarType(0.5)*(k1 + k2*uh*uh)*ufl.inner(ufl.grad(uh), ufl.grad(uh))*ufl.dx - f*uh*ufl.dx
J = dl.fem.form(Jform)

# ---- the gradient of the energy functional
u_tilde = ufl.TestFunction(Vh)
gradform = (k2*uh*u_tilde)*ufl.inner(ufl.grad(uh), ufl.grad(uh))*ufl.dx + \
       (k1 + k2*uh*uh)*ufl.inner(ufl.grad(uh), ufl.grad(u_tilde))*ufl.dx - f*u_tilde*ufl.dx
grad = dl.fem.form(gradform)

# ---- prepare data for finite-difference check
# ---- |[J(u0 + eps udir) - J(u0)] / eps - grad(J)^T udir| = O(eps) 

u0 = dl.fem.Function(Vh)
u0.interpolate(lambda x: x[0]*(x[0]-1)*x[1]*(x[1]-1))

uh.vector.zeroEntries()
uh.vector.axpy(1.0, u0.vector)
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

J0 = mesh.comm.allreduce(dl.fem.assemble_scalar(J), op = MPI.SUM)
grad0 = dl.fem.petsc.assemble_vector(grad)
grad0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
grad0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dl.fem.petsc.set_bc(grad0, bcs)

u_dir =  dl.la.create_petsc_vector(Vh.dofmap.index_map, Vh.dofmap.index_map_bs)
u_dir.set(1.0)
u_dir.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dl.fem.petsc.set_bc(u_dir, bcs)

grad0Tudir = grad0.dot(u_dir)
n_eps = 32
epss = 1e-2*np.power(2., -np.arange(n_eps))
fdgrad_residual = np.zeros(n_eps)

for i, eps in enumerate(epss):
    uh.vector.scale(0.0)
    uh.vector.axpy(1.0, u0.vector)
    uh.vector.axpy(eps, u_dir) # uh = uh + eps[i]*dir
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    Jplus = mesh.comm.allreduce(dl.fem.assemble_scalar(J), op = MPI.SUM)
    fdgrad_residual[i] = abs( (Jplus - J0)/eps - grad0Tudir )

if iAmRoot:
    plt.loglog(epss, fdgrad_residual, "-ob", label="Error Grad")
    plt.loglog(epss, (.5*fdgrad_residual[0]/epss[0])*epss, "-.k", label=r"First Order, $\propto\varepsilon$")
    plt.title("Finite difference check of the first variation")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$r_{1}(\varepsilon)$, finite-difference error")
    plt.legend(loc = "upper left")
    plt.show()

u_hat   = ufl.TrialFunction(Vh)
Hform = k2*u_hat*u_tilde*ufl.inner(ufl.grad(uh), ufl.grad(uh))*ufl.dx + \
     PETSc.ScalarType(2.0)*(k2*uh*u_tilde)*ufl.inner(ufl.grad(u_hat), ufl.grad(uh))*ufl.dx + \
     PETSc.ScalarType(2.0)*k2*u_hat*uh*ufl.inner(ufl.grad(uh), ufl.grad(u_tilde))*ufl.dx + \
     (k1 + k2*uh*uh)*ufl.inner(ufl.grad(u_tilde), ufl.grad(u_hat))*ufl.dx
H = dl.fem.form(Hform)

uh.vector.zeroEntries()
uh.vector.axpy(1.0, u0.vector)
uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

H_0 = dl.fem.petsc.assemble_matrix(H, [])
H_0.assemble()

H_0udir = dl.fem.Function(Vh)
diff_grad = dl.fem.Function(Vh)


H_0.mult(u_dir, H_0udir.vector)
dl.fem.petsc.set_bc(H_0udir.vector, bcs)
H_0udir.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
fdHessian_residuals = np.zeros(n_eps)

for i, eps in enumerate(epss):
    uh.vector.zeroEntries()
    uh.vector.axpy(1.0, u0.vector)
    uh.vector.axpy(eps, u_dir)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    grad_plus = dl.fem.petsc.assemble_vector(grad)
    grad_plus.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    grad_plus.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dl.fem.petsc.set_bc(grad_plus, bcs)

    diff_grad.vector.zeroEntries()
    diff_grad.vector.axpy(1., grad_plus)
    diff_grad.vector.axpy(-1., grad0)
    diff_grad.vector.scale(1./eps)
    diff_grad.vector.axpy(-1, H_0udir.vector)
    diff_grad.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    fdHessian_residuals[i] = diff_grad.vector.norm(2) 


if iAmRoot:
   plt.figure()    
   plt.loglog(epss, fdHessian_residuals, "-ob", label="Error Hessian")
   plt.loglog(epss, (.5*fdHessian_residuals[0]/epss[0])*epss, "-.k", label=r"First Order, $\propto\varepsilon$")
   plt.title("Finite difference check of the second variation")
   plt.xlabel(r"$\varepsilon$")
   plt.ylabel(r"$r_{2}(\varepsilon)$, finite-difference error")
   plt.legend(loc = "upper left")
   plt.show()


uh.vector.zeroEntries()
rgradtol = 1.e-9
max_iter = 14
total_cg_its = 0
J0 = mesh.comm.allreduce(dl.fem.assemble_scalar(J), op = MPI.SUM)
g0 = dl.fem.petsc.assemble_vector(grad)
g0.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
g0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dl.fem.petsc.set_bc(g0, bcs)


grad0norm = g0.norm(2)
tol = grad0norm * rgradtol
uhat = dl.fem.Function(Vh)

if iAmRoot:
    print ("{0:3} {1:3}  {2:15} {3:15} {4:15}".format(
      "It", "cg its", "    J    ", "(g, uhat)", "||g||l2") )

for i in range(max_iter):
    Hn = dl.fem.petsc.assemble_matrix(H, bcs)
    Hn.assemble()
    gn = dl.fem.petsc.assemble_vector(grad)
    gn.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    gn.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dl.fem.petsc.set_bc(gn, bcs)
    gradnorm = gn.norm(2)
    if gradnorm < tol:
        if iAmRoot:
            print("\nNorm of the gradient at minimizer estimate", gradnorm )
            print("Value of the energy functional at minimizer estimate", Ji)
            print("Converged in {0:d} Newton iterations and {1:d} total conjugate-gradient iterations.".format(i, total_cg_its))
        break
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = min(0.5, np.sqrt(gradnorm/grad0norm))
    opts["pc_type"]  = "gamg"
    # Create PETSc Krylov solver
    solver = PETSc.KSP().create(mesh.comm)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(Hn)
    
    # Solve linear system Hn uhat = -gn
    solver.solve(-gn, uhat.vector)
    uhat.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    # record number of CG-iterations for linear system solution
    cg_its = solver.its
    total_cg_its += cg_its
    # uh = uh + uhat (better to use backtracking line search for robustness)
    uh.vector.axpy(1.0, uhat.vector)
    uh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    
    
    
    Ji = mesh.comm.allreduce(dl.fem.assemble_scalar(J), op = MPI.SUM)
    guhat = gn.dot(uhat.vector)
    gnorm = gn.norm(2)
    if iAmRoot:
        print ("{0:3d} {1:2d}       {2:1.2e}      {3:1.2e}       {4:1.2e}".format(
           i, cg_its, Ji, -guhat, gradnorm) )   


grid.point_data["u"] = uh.x.array.real
plotter = pv.Plotter(window_size=window_size)
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="u")
plotter.view_xy()
plotter.add_title('Optimizer', font='courier', color='w', font_size=10)
plotter.background_color = "pink"
plotter.show()
plotter.close()

