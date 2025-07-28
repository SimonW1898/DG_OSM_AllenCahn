from modules import Mesh, SubMesh, DG, DomainDecomposition
import numpy as np
import plot_radius as pr

# solve Allen Cahn with DG
N = 16
dof = 1
domain = [-0.5,0.5]
print("Constructing mesh")
puremesh = Mesh(cell_shape=(N,N), dof=dof, x_domain=domain, y_domain=domain)
# create a dg object

penalty = 18
symmetry = -1 
superpenalisation = 1
linear_solver_tol = 1e-14
alpha = 1
beta = 0
dg = DG(mesh = puremesh, penalty = penalty, symmetry_parameter = symmetry,\
        superpenalisation = superpenalisation, linear_solver_tol = linear_solver_tol, \
        alpha = alpha, beta = beta)

# solve the dg allen cahn
print("Solving Allen-Cahn")
radius_path = dg.solve_allen_cahn()
pr.plot_radius_from_file(radius_path)
# plot the solution
dg.mesh.plot_all_solutions()


# domain decomposition
number_of_elements = 16
number_of_subdomains = 2
overlap = 1
dof = 2
domain = [-0.5,0.5]
alpha = 1
beta = 1
symmetry = -1
penalty = 40
superpenalisation = False
max_iter = 10
linear_solver_tol = 1e-14
dd_tol = 1e-13


dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = number_of_subdomains,\
                        overlap = overlap, dof = dof, domain = domain,  \
                        alpha = alpha, beta=beta, \
                        symmetry_parameter = symmetry, penalty = penalty, superpenalisation = superpenalisation, 
                        linear_solver_tol = linear_solver_tol)


dd.global_mesh.plot_solution()

radius_dat_path = dd.solve_allen_cahn_dd(max_iter = max_iter, max_residual = dd_tol, modulo_plot = 10)

pr.plot_radius_from_file_dd(radius_dat_path)