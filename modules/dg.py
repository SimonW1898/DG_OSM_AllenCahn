# packages
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt

import pandas as pd


from .mesh import Mesh
from .submesh import SubMesh
from .quadrature import Quadrature
from .basis_functions import BasisFunction

import datetime
import os

class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



class DG():
    """  
    Solve the Poisson equation using the Discontinuous Galerkin method
    for the given mesh
    """
    def __init__(self,mesh, penalty = 18, symmetry_parameter = -1, superpenalisation = 1,linear_solver_tol = 1e-14, alpha = 1, beta = 0):
        self.mesh = mesh
        self.penalty = penalty
        self.symmetry = symmetry_parameter
        self.superpenalisation = penalty/(self.mesh.hx)**superpenalisation
        self.superpenalisation_parameter = superpenalisation
        # create quadrature object
        self.quadrature = Quadrature(degree=self.mesh.dof+1)
        # create basis function object
        self.basis = BasisFunction(degree=self.mesh.dof,quadrature=self.quadrature)

        # linear solver tolerance
        self.linear_solver_tol = linear_solver_tol
        # extension to robin boundary condition
        self.alpha = alpha
        self.beta = beta

        self.is_robinBC = not ((alpha, beta) in [(1, 0), (0, 1)])
        self.split_average = False

        # set alpha and beta for exterior boundary
        self.alpha_exterior = 1
        self.beta_exterior = 0
        self.is_robinBC_exterior = not((self.alpha_exterior, self.beta_exterior) in [(1, 0), (0, 1)])
        # if self.is_robinBC_exterior:
        #     print("WARNING: Exterior BC should be pure neumann or dirichlet")


        # create Jacobians
        # 2D Jacobian
        self.Jacobian = self.compute_jacobian()
        self.Jacobian_inv = np.linalg.inv(self.Jacobian)
        self.Jacobian_det = np.linalg.det(self.Jacobian)
        # 1D Jacobian
        self.Jacobian1D = self.compute_jacobian1D()
        self.Jacobian1D_inv = 1/(self.Jacobian1D)
        self.Jacobian1D_det = self.Jacobian1D



        # for solving
        self.A = self.build_global_matrix()
        self.set_linear_operator()

        self.mesh.set_rhs(self.compute_rhs())
        # self.solve_poisson_matrix_free()
        # self.compute_error_with_overlap()
        # if self.mesh.parent_mesh is not None:
        #     self.compute_error_without_overlap()
        

    # Jacobians
    def compute_jacobian(self):
        """ 
        because of quadratic elements the Jacobian is constant
        diagonal matrix with hx and hy
        Output:
            (np.array) 2x2 Jacobian matrix
        """
        return np.diag([self.mesh.hx,self.mesh.hy])
    
    def compute_jacobian1D(self):
        """ 
        compute the Jacobian for the 1D integral
        Output:
            (np.array) 2x1 Jacobian array [0] for x and [1] for y
        """
        return self.mesh.hx
    
    # Source function
    def source(self,x):
        """ 
        Compute the right-hand side of the weak formulation/ source term
        Input:
            x: n x 2 (np.array)  array of the coordinates of the quadrature points
        Output:
            f: n (np.array) right-hand side vector
        """
        x_coords = x[:, 0]
        y_coords = x[:, 1]
        
        # Original terms with x and y as provided
        term1 = 2 * np.exp(-x_coords**2 - y_coords**2)
        term2 = (1 + 3*x_coords - 5*x_coords**2 - 2*x_coords**3 + 2*x_coords**4)
        term3 = (-1 + y_coords) * y_coords
        
        # Interchanged terms with x and y swapped
        term1_swapped = 2 * np.exp(-y_coords**2 - x_coords**2)
        term2_swapped = (1 + 3*y_coords - 5*y_coords**2 - 2*y_coords**3 + 2*y_coords**4)
        term3_swapped = (-1 + x_coords) * x_coords
        
        # Compute the function values
        f_original = term1 * term2 * term3
        f_swapped = term1_swapped * term2_swapped * term3_swapped
        
        # Return the sum of the original and swapped function values
        f = f_original + f_swapped
        
        return - f
    

    # Quadrature points
    def get_global_quadrature_points(self, elem_idx):
        """   
        Compute the global coordinates of the quadrature points
        Input:
            elem_idx: int element index
        Output:
            x: N_quad x 2 (np.array) global coordinates of the quadrature points
        """
        # global points from the reference element to global coordinates
        x_ref2glob = np.matmul(self.quadrature.points, self.Jacobian)
        # get the coordinates of the bottom left node of the element
        global_node_idx = self.mesh.element_connectivity[elem_idx, 0]
        # get the global coordinates of the bottom left node of the element
        global_bottom_left_node_coord = self.mesh.coord[global_node_idx]
        # add the global coordinates of the bottom left node to the global quadrature points of the reference element
        x = x_ref2glob + global_bottom_left_node_coord
        return x
    

    def compute_local_mass_matrix(self):
        """  
        Compute the local mass matrix
        M_ij = (phi_i, phi_j)
        Output:
            (np.array) (dof+1)^2 x (dof+1)^2 local mass matrix
        """
        M = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))
        # loop over basis functions i,j
        for i in range(self.mesh.N_dofs):
                for j in range(self.mesh.N_dofs):
                    # compute the volume integral
                    M[i,j] = self.Jacobian_det * np.dot(self.quadrature.weights,\
                                self.basis.basis_vals[:,i] * self.basis.basis_vals[:,j])
        return M
    
    def compute_local_stiffness_matrix(self, elem_idx):
        """  
        Compute the local stiffness matrix
        S_ij = (grad phi_i, grad phi_j)
        Input:
            elem_idx: int element index
        Output:
            (np.array) (dof+1)^2 x (dof+1)^2 local stiffness matrix
        """
        S = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))
        
        # loop over quadrature points
        for k in range(self.quadrature.N_quad):
            factor = self.quadrature.weights[k] * self.Jacobian_det
            # loop over basis functions i,j
            for i in range(self.mesh.N_dofs):
                for j in range(self.mesh.N_dofs):
                    # compute the volume integral
                    S[i,j] += factor * \
                            np.dot(np.dot(self.Jacobian_inv,self.basis.basis_der_vals[k,i,:]),\
                                    np.dot(self.Jacobian_inv, self.basis.basis_der_vals[k,j,:]))
                    
        return S
    
    def compute_rhs_volume_contribution(self, elem_idx):
        """  
        Compute the volume contribution to the right-hand side
        Input:
            elem_idx: int element index
        Output:
            (np.array) N_dof right-hand side vector
        """
        rhs = np.zeros(self.mesh.N_dofs)
        # compute values on the quadrature points
        x_glob_quad = self.get_global_quadrature_points(elem_idx)
        f = self.source(x_glob_quad)
        # loop over quadrature points
        for k in range(self.quadrature.N_quad):
            factor = self.quadrature.weights[k] * self.Jacobian_det
            # loop over basis functions i
            for i in range(self.mesh.N_dofs):
                # compute the volume integral
                rhs[i] += factor * f[k] * self.basis.basis_vals[k,i]
       
        return rhs

    def compute_local_interior_face(self, face_idx):
        """ 
        Compute the local interior face integral
        All faces that are interior of the local mesh
        Input:
            face_idx: int face index
        Output:
            M11: N_dofs x N_dofs matrix E1 x E1
            M12: N_dofs x N_dofs matrix E1 x E2
            M21: N_dofs x N_dofs matrix E2 x E1
            M22: N_dofs x N_dofs matrix E2 x E2
        """
        # initialize matrices
        M11 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))
        M12 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))
        M21 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))
        M22 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))


        # get the local face indices
        if self.mesh.is_vertical_face[face_idx]:
            face_idx_E1 = 1
            face_idx_E2 = 0
        else:
            face_idx_E1 = 3
            face_idx_E2 = 2

        # get the values of the basis functions on the face quadrature points
        basis_vals_face = self.basis.basis_vals_face_integral
        basis_der_vals_face = self.basis.basis_der_vals_face_integral

        # get the values on the face quadrature points
        basis_vals_face_E1 = basis_vals_face[face_idx_E1,:,:]
        basis_vals_face_E2 = basis_vals_face[face_idx_E2,:,:]
        # derivative values on the face quadrature points
        basis_der_vals_face_E1 = basis_der_vals_face[face_idx_E1,:,:,:]
        basis_der_vals_face_E2 = basis_der_vals_face[face_idx_E2,:,:,:]

        # reshape into N_quad_1D x N_Nodes_1D x 2(dim)  
        basis_der_vals_face_E1 = np.transpose(basis_der_vals_face_E1,(1,2,0))
        basis_der_vals_face_E2 = np.transpose(basis_der_vals_face_E2,(1,2,0))
        
        # get the normal vector
        normal_vector = self.mesh.normal_vectors[face_idx_E1]

        # loop over quadrature points
        for k in range(self.quadrature.Nquad_1D):
            # factor that appears when discretizing the volume integral
            factor = 0.5*self.quadrature.weights_1D[k]  #* self.Jacobian_1D #  (1/self.Jacobian_1D)**2 * self.det_Jacobian_1D two times inverse Jacobian
            # compute M11_k contribution
            for i in range(self.mesh.N_dofs):
                for j in range(self.mesh.N_dofs):
                    M11[i,j] += -1            * factor* basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector)\
                        + self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector)\
                        + self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
            
            # compute M22_k contribution (same as M11)
                    M22[i,j] +=  1 *                 factor * basis_vals_face_E2[k,i] * np.dot(basis_der_vals_face_E2[k,j,:], normal_vector)\
                        -1 * self.symmetry * factor * basis_vals_face_E2[k,j] * np.dot(basis_der_vals_face_E2[k,i,:], normal_vector)\
                        +    self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] * basis_vals_face_E2[k,i] * basis_vals_face_E2[k,j] 

            # compute M12_k contribution
                    M12[i,j] += -1                 * factor * basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E2[k,j,:], normal_vector) \
                        -1 * self.symmetry * factor * basis_vals_face_E2[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector)\
                        -1 * self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] * basis_vals_face_E1[k,i] * basis_vals_face_E2[k,j]

            # compute M21_k contribution
                    M21[i,j] +=                      factor * basis_vals_face_E2[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector)\
                        +   self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E2[k,i,:], normal_vector)\
                        -1 * self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] * basis_vals_face_E2[k,i] * basis_vals_face_E1[k,j]
        
        return M11, M12, M21, M22

    def get_face_quadrature_points(self, elem_idx, face_idx):
        """
        Compute the global coordinates of the quadrature points on the face
        """
    # make quadrature points on face global in 2d make case operator for 4 face indices
        # bottom left index of the element
        bottom_left_coord = self.mesh.coord[self.mesh.bottom_left_index[elem_idx]]
        x = np.zeros((self.quadrature.points_1D.shape[0],2))
        if face_idx == 0:
            x[:,0] = bottom_left_coord[0]
            x[:,1] = self.Jacobian1D * self.quadrature.points_1D + bottom_left_coord[1]
        elif face_idx == 1:
            x[:,0] = bottom_left_coord[0] + self.mesh.hx
            x[:,1] = self.Jacobian1D * self.quadrature.points_1D + bottom_left_coord[1]
        elif face_idx == 2:
            x[:,0] = self.Jacobian1D * self.quadrature.points_1D + bottom_left_coord[0]
            x[:,1] = bottom_left_coord[1]
        elif face_idx == 3:
            x[:,0] = self.Jacobian1D * self.quadrature.points_1D + bottom_left_coord[0]
            x[:,1] = bottom_left_coord[1] + self.mesh.hy
        return x
    
    def compute_local_exterior_boundary_face(self, face_idx, boundary_side):
        """   
        Compute the local boundary face integral
        Exterior boundary of the parent mesh
        Input: 
            face_idx: int face index
            boundary_side: int boundary side index (0: left, 1: right, 2: bottom, 3: top)
        Output:
            M11: N_dofs x N_dofs matrix E1 x E1
        """    
        # initialize matrix
        M11 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))


        # get the local face indices (0,1,2,3) 
        if boundary_side == 0:
            face_idx_E1 = 0
            face_idx_E2 = 1
        elif boundary_side == 1:
            face_idx_E1 = 1
            face_idx_E2 = 0
        elif boundary_side == 2:
            face_idx_E1 = 2
            face_idx_E2 = 3
        elif boundary_side == 3:
            face_idx_E1 = 3
            face_idx_E2 = 2

        # get the element index
        neighbour_indices = self.mesh.face_neighbours[face_idx]
        elem_idx = neighbour_indices[0] if neighbour_indices[0] != -1 else neighbour_indices[1]

        # get normal vector
        normal_vector = self.mesh.normal_vectors[face_idx_E1]

        # get the values of the basis functions on the face quadrature points
        basis_vals_face = self.basis.basis_vals_face_integral
        basis_der_vals_face = self.basis.basis_der_vals_face_integral

        # basis function values on the face
        basis_vals_face_E1 = basis_vals_face[face_idx_E1,:,:]
        # basis function derivatives on the face
        basis_der_vals_face_E1 = basis_der_vals_face[face_idx_E1,:,:,:]

        # reshape into N_quad_1D x N_Nodes_1D x 2(dim)  
        basis_der_vals_face_E1 = np.transpose(basis_der_vals_face_E1,(1,2,0))

        # loop over quadrature points
        for k in range(self.quadrature.Nquad_1D):
            # factor that appears when discretizing the volume integral
            factor = self.quadrature.weights_1D[k]
            # loop over the basis functions i,j
            for i in range(self.mesh.N_dofs):
                for j in range(self.mesh.N_dofs):
                    if self.is_robinBC_exterior:
                        if self.split_average:
                            M11[i,j] += -1 * 0.5            * factor* basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector)\
                                + self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector) \
                                        + self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] \
                                                    * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
                        else:
                            M11[i,j] += self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector) \
                                    + self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] \
                                                * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
                        

                    else:    
                          M11[i,j] += -1            * factor* basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector) \
                        + self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector) \
                        + self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] \
                              * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
        return M11
    
    def compute_rhs_exterior_boundary_face_contribution(self, face_idx, boundary_side):
        """   
        Compute rhs contribution of the exterior boundary face
        Input:
            face_idx: int face index
            boundary_side: int boundary side index (0: left, 1: right, 2: bottom, 3: top)
        Output:
            rhs: N_dofs (np.array) right-hand side vector
        """
        # initialize right-hand side vector
        rhs = np.zeros(self.mesh.N_dofs)
        # get the local face indices (0,1,2,3) 
        if boundary_side == 0:
            face_idx_E1 = 0
            face_idx_E2 = 1
        elif boundary_side == 1:
            face_idx_E1 = 1
            face_idx_E2 = 0
        elif boundary_side == 2:
            face_idx_E1 = 2
            face_idx_E2 = 3
        elif boundary_side == 3:
            face_idx_E1 = 3
            face_idx_E2 = 2

        # get the element index
        neighbour_indices = self.mesh.face_neighbours[face_idx]
        elem_idx = neighbour_indices[0] if neighbour_indices[0] != -1 else neighbour_indices[1]

        # get normal vector
        normal_vector = self.mesh.normal_vectors[face_idx_E1]

        # get the values of the basis functions on the face quadrature points
        basis_vals_face = self.basis.basis_vals_face_integral
        basis_der_vals_face = self.basis.basis_der_vals_face_integral

        # basis function values on the face
        basis_vals_face_E1 = basis_vals_face[face_idx_E1,:,:]
        # basis function derivatives on the face
        basis_der_vals_face_E1 = basis_der_vals_face[face_idx_E1,:,:,:]

        # reshape into N_quad_1D x N_Nodes_1D x 2(dim)  
        basis_der_vals_face_E1 = np.transpose(basis_der_vals_face_E1,(1,2,0))

        # get the boundary values on the face quadrature points
        x_glob = self.get_face_quadrature_points(elem_idx, face_idx_E1)

        # get the values on the face quadrature points of the boundary condition
        f =  self.mesh.analytical_solution(x_glob)
        f_grad = self.mesh.grad_analytical_solution(x_glob)

        if self.is_robinBC_exterior:
            f = self.alpha_exterior * f
            f_grad = self.beta_exterior * f_grad

        for k in range(self.quadrature.Nquad_1D):
            # factor that appears when discretizing the volume integral
            factor = self.quadrature.weights_1D[k]
            # loop over the basis functions i
            for i in range(self.mesh.N_dofs):
                if self.is_robinBC_exterior:
                    rhs[i] += self.quadrature.weights_1D[k] * f[k] * self.Jacobian1D_det* \
                            (self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:],normal_vector) + \
                            self.superpenalisation * basis_vals_face_E1[k,i]) \

                    + \
                            self.quadrature.weights_1D[k] * self.Jacobian1D_det * \
                            np.dot(f_grad[k], normal_vector) * basis_vals_face_E1[k,i]
                else:
                    rhs[i] += self.alpha_exterior * \
                                self.quadrature.weights_1D[k] * f[k] * self.Jacobian1D_det* \
                                    (self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:],normal_vector) + \
                                    self.superpenalisation * basis_vals_face_E1[k,i]) + \
                            self.beta_exterior * \
                                self.quadrature.weights_1D[k] * np.dot(f_grad[k], normal_vector) * self.Jacobian1D_det * \
                                    basis_vals_face_E1[k,i]
                    
            
        return rhs
    
    def plot_rhs(self,rhs):
        """
        for testing plot rhs and store it
        """
        fname = "/wsgjsc/home/wenchel1/dg_nasm/output/plots/rhs.png"
        plt.plot(rhs)
        plt.savefig(fname)
        plt.close()
        print("rhs plot saved at: ", fname)



    # to handle interior boundaries of the submesh
    def get_values_on_interior_boundary_quadrature_points(self, face_idx_E2, global_elem_nodes_E2, elem_idx = 0, face_idx_E1 = 0):
        """ 
        Get values on the quadrature points on the face of the ghost cell 
        Needed to compute dirichlet BC for interior boundaries

        """
        values = np.zeros((self.quadrature.Nquad_1D))
        # get the values of the boundary condition
        

        # get the boundary values on the face quadrature points
        # g_D = self.mesh.parent_mesh.u[global_face_nodes_E2].squeeze()
        g_D = self.mesh.parent_mesh.u[global_elem_nodes_E2].squeeze()

        x_glob = self.get_face_quadrature_points(elem_idx, face_idx_E1)
        g_D_analytical = self.mesh.analytical_solution(x_glob)

        # get the local face indices with face_idx_E2 to loop over the basis functions
        # local_face_nodes = self.mesh.local_faces[face_idx_E2]

        # loop over basis functions and compute the values on the quadrature points
        for k in range(self.quadrature.Nquad_1D):
                values[k] = np.dot(g_D,self.basis.basis_vals_face_integral[face_idx_E2,k,:])
                
        return values

    def get_gradient_values_on_interior_boundary_quadrature_points(self, face_idx_E2, global_elem_nodes_E2, elem_idx = 0, face_idx_E1 = 0):
        """ 
        Get values on the quadrature points on the face of the ghost cell 
        Needed to compute dirichlet BC for interior boundaries
        """
        values_gradient = np.zeros((self.quadrature.Nquad_1D,2))
        values = np.zeros((self.quadrature.Nquad_1D))
        # get the values of the boundary condition
        g_D = self.mesh.parent_mesh.u[global_elem_nodes_E2].squeeze()
        # loop over basis functions and compute the values on the quadrature points
        for k in range(self.quadrature.Nquad_1D):
            # values[k] = np.dot(g_D,self.basis.basis_vals_face_integral[face_idx_E2,k,:])
            values_gradient[k,0] = np.dot(g_D,self.basis.basis_der_vals_face_integral[face_idx_E2,0,k,:])
            values_gradient[k,1] = np.dot(g_D,self.basis.basis_der_vals_face_integral[face_idx_E2,1,k,:])

        values_gradient = np.dot(values_gradient, self.Jacobian1D_inv)
        return values_gradient

    
    def compute_local_interior_boundary_face(self, face_idx, boundary_side):
        """   
        Compute the boundary face integral for interior boundaries
        Input:
            face_idx: int face index
            boundary_side: int boundary side index (0: left, 1: right, 2: bottom, 3: top)
        Output:
            M11: N_dofs x N_dofs matrix E1 x E1
        """
        # initialize matrix
        M11 = np.zeros((self.mesh.N_dofs,self.mesh.N_dofs))

        # get the local face indices (0,1,2,3) 
        if boundary_side == 0:
            face_idx_E1 = 0
            face_idx_E2 = 1
        elif boundary_side == 1:
            face_idx_E1 = 1
            face_idx_E2 = 0
        elif boundary_side == 2:
            face_idx_E1 = 2
            face_idx_E2 = 3
        elif boundary_side == 3:
            face_idx_E1 = 3
            face_idx_E2 = 2

        # get the element index
        neighbour_indices = self.mesh.face_neighbours[face_idx]
        elem_idx = neighbour_indices[0] if neighbour_indices[0] != -1 else neighbour_indices[1]

        # get normal vector
        normal_vector = self.mesh.normal_vectors[face_idx_E1]

        # get the values of the basis functions on the face quadrature points
        basis_vals_face = self.basis.basis_vals_face_integral
        basis_der_vals_face = self.basis.basis_der_vals_face_integral

        # basis function values on the face
        basis_vals_face_E1 = basis_vals_face[face_idx_E1,:,:]
        # basis function derivatives on the face
        basis_der_vals_face_E1 = basis_der_vals_face[face_idx_E1,:,:,:]

        # reshape into N_quad_1D x N_Nodes_1D x 2(dim)  
        basis_der_vals_face_E1 = np.transpose(basis_der_vals_face_E1,(1,2,0))

        # loop over quadrature points
        for k in range(self.quadrature.Nquad_1D):
            # factor that appears when discretizing the volume integral
            factor = self.quadrature.weights_1D[k]
            # loop over the basis functions i,j
            for i in range(self.mesh.N_dofs):
                for j in range(self.mesh.N_dofs):
                    if self.is_robinBC:
                        M11[i,j] +=      self.Jacobian1D_det * self.quadrature.weights_1D[k]*(
                                - 0.5 * self.Jacobian1D_inv * basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector) \
                                + self.alpha* self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector) * basis_vals_face_E1[k,j] \
                                + self.alpha * self.superpenalisation * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
                            )
                                    
                    else:
                        M11[i,j] += -1            * factor* basis_vals_face_E1[k,i] * np.dot(basis_der_vals_face_E1[k,j,:], normal_vector) \
                        + self.symmetry * factor * basis_vals_face_E1[k,j] * np.dot(basis_der_vals_face_E1[k,i,:], normal_vector) \
                        + self.superpenalisation * self.Jacobian1D_det * self.quadrature.weights_1D[k] \
                                                * basis_vals_face_E1[k,i] * basis_vals_face_E1[k,j]
        return M11

    def compute_rhs_interior_boundary_face(self, face_idx, boundary_side):
            """   
            Compute the boundary face integral for interior boundaries
            Input:
                face_idx: int face index
                boundary_side: int boundary side index (0: left, 1: right, 2: bottom, 3: top)
            Output:
                M11: N_dofs x N_dofs matrix E1 x E1
            """
            # initialize matrix
            rhs = np.zeros(self.mesh.N_dofs)

            # get the local face indices (0,1,2,3) 
            if boundary_side == 0:
                face_idx_E1 = 0
                face_idx_E2 = 1
            elif boundary_side == 1:
                face_idx_E1 = 1
                face_idx_E2 = 0
            elif boundary_side == 2:
                face_idx_E1 = 2
                face_idx_E2 = 3
            elif boundary_side == 3:
                face_idx_E1 = 3
                face_idx_E2 = 2

            # get the element index
            neighbour_indices = self.mesh.face_neighbours[face_idx]
            elem_idx = neighbour_indices[0] if neighbour_indices[0] != -1 else neighbour_indices[1]

            # get normal vector
            normal_vector = self.mesh.normal_vectors[face_idx_E1]
            normal_vector_E2 = self.mesh.normal_vectors[face_idx_E2]

            # get the values of the basis functions on the face quadrature points
            basis_vals_face = self.basis.basis_vals_face_integral
            basis_der_vals_face = self.basis.basis_der_vals_face_integral

            # basis function values on the face
            basis_vals_face_E1 = basis_vals_face[face_idx_E1,:,:]
            # basis function derivatives on the face
            basis_der_vals_face_E1 = basis_der_vals_face[face_idx_E1,:,:,:]

            # reshape into N_quad_1D x N_Nodes_1D x 2(dim)  
            basis_der_vals_face_E1 = np.transpose(basis_der_vals_face_E1,(1,2,0))

            # get global global_face_nodes_E2
            # get the global element index of E1
            global_face_idx_E1 = self.mesh.mapping_subdomain2global_elements.flatten()[elem_idx]

            # get the global element index of E2 (ghost cell)
            global_face_idx_E2 = self.mesh.parent_mesh.element_neighbours[global_face_idx_E1,face_idx_E1]
            # get the element index of E2 (ghost cell)
            global_face_nodes_E2 = self.mesh.parent_mesh.get_face_indices(global_face_idx_E2)[face_idx_E2]
            global_elem_nodes_E2 = self.mesh.parent_mesh.element_connectivity[global_face_idx_E2]
            # get the values on the face quadrature points of the boundary condition
            f =  self.get_values_on_interior_boundary_quadrature_points(face_idx_E2, global_elem_nodes_E2, elem_idx, face_idx_E1)
            # print("face_idx", face_idx)
            # print("normal vector", normal_vector_E2)
            f_grad =  self.get_gradient_values_on_interior_boundary_quadrature_points(face_idx_E2, global_elem_nodes_E2, elem_idx, face_idx_E1)

            if self.is_robinBC:
                f = self.alpha * f
                f_grad = self.beta * f_grad

            # loop over quadrature points
            for k in range(self.quadrature.Nquad_1D):
                # loop over the basis functions i,j
                for i in range(self.mesh.N_dofs):
                    if self.is_robinBC:
                        if self.split_average:
                            rhs[i] += self.quadrature.weights_1D[k] * self.Jacobian1D_det* \
                            (0.5 *  basis_vals_face_E1[k,i] * np.dot(f_grad[k], normal_vector) \
                            + self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:],normal_vector) * f[k] \
                            + self.superpenalisation * basis_vals_face_E1[k,i] * f[k])
                        else:
                            rhs[i] += self.quadrature.weights_1D[k] * f[k] * self.Jacobian1D_det* \
                                    (self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:],normal_vector) + \
                                    self.superpenalisation * basis_vals_face_E1[k,i]) \
                            + \
                                    self.quadrature.weights_1D[k] * self.Jacobian1D_det * \
                                    np.dot(f_grad[k], normal_vector) * basis_vals_face_E1[k,i]
                            
                    else:
                        rhs[i] += self.alpha * \
                                    self.quadrature.weights_1D[k] * f[k] * self.Jacobian1D_det* \
                                        (self.symmetry * self.Jacobian1D_inv * np.dot(basis_der_vals_face_E1[k,i,:],normal_vector) + \
                                        self.superpenalisation * basis_vals_face_E1[k,i]) + \
                                self.beta * \
                                 self.quadrature.weights_1D[k] * self.Jacobian1D_det * \
                                    np.dot(f_grad[k,:], normal_vector) * basis_vals_face_E1[k,i]
                    
            return rhs

    def compute_rhs(self):
        """  
        Compute the global right-hand side vector
        loop over elements and boundary faces (interior faces do not contribute to the right-hand side)
        Output:
            rhs: N_dofs (np.array) right-hand side vector
        """
        # initialize rhs vector
        rhs = np.zeros(self.mesh.N_nodes)
        rhs_dg = np.zeros(self.mesh.N_nodes)
        rhs_mass = np.zeros(self.mesh.N_nodes)
        # loop over faces
        # boundary_side/face 0 and 2 have global indices at position 1
        # boundary_side/face 1 and 3 have global indices at position 0
        node_position = [1,0,1,0]
        # rhs_volume = rhs.copy()
        # plt.plot(rhs, label = "rhs")
        # loop over boundary faces and check if theyre exterior or interior boundary
        for boundary_side in range (4):
            #exterior 
            if self.mesh.is_exterior_boundary[boundary_side]:
                # loop over boundary faces
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # get element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the node indices of the face
                    idx_E1 = self.mesh.element_connectivity[elem_idx]
                    # compute local rhs
                    rhs_dg[idx_E1] += self.compute_rhs_exterior_boundary_face_contribution(face_idx, boundary_side)
                
            #interior
            else:
                # print("boundary side:", boundary_side)
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # get element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the node indices of the face
                    idx_E1 = self.mesh.element_connectivity[elem_idx]
                    # compute local rhs
                    rhs_dg[idx_E1] +=  self.compute_rhs_interior_boundary_face(face_idx, boundary_side)

        # compute the rhs contribution of mass matrix 
        M_loc = self.compute_local_mass_matrix()
        for elem_idx in range(self.mesh.N_elems):
            idx = self.mesh.element_connectivity[elem_idx]
            rhs_mass[idx] += np.dot(M_loc,self.mesh.u[idx])
        # now the nonlinear part of the rhs needs to be added
        rhs_nl = self.compute_nonlinear_rhs()
        rhs = rhs_nl + rhs_mass + self.mesh.K_ac * self.mesh.dt * rhs_dg
        return rhs

    
    def compute_nonlinear_rhs(self, u = None):
        """
        compute the nonlinear rhs contribution by setting u(n+1) = u(n)
        loop over the elements and compute (f(phi_j u_j), phi_i)
        """
        if u is None:
            u = self.mesh.u

        rhs_nonlinear = np.zeros(self.mesh.N_nodes)
        # loop over elements
        for elem_idx in range(self.mesh.N_elems):
            # global node indices of the element
            node_idx = self.mesh.element_connectivity[elem_idx]
            # compute the local nonlinear rhs contribution sum_j f(phi_j u_j) 
            f = self.f_nonlinear_local(u[node_idx])
            for k in range(self.quadrature.N_quad):
                factor = self.quadrature.weights[k] * self.Jacobian_det
                # compute rhs contribution rhs_loc[i] =  f * phi_i
                for i in range(self.mesh.N_dofs):
                    #dot product is like looping over quadrature the quadrature weights are already accounted for in f
                    rhs_nonlinear[node_idx[i]] += factor * f[k] * self.basis.basis_vals[k,i]

        nonlinear_factor = - self.mesh.K_ac * self.mesh.dt /(self.mesh.epsilon_ac**2)
        rhs_nonlinear = nonlinear_factor * rhs_nonlinear
        return rhs_nonlinear
    
    def test_rhs(self, u = None):
        """
        compute the nonlinear rhs contribution by setting u(n+1) = u(n)
        loop over the elements and compute (f(phi_j u_j), phi_i)
        """
        if u is None:
            u = self.mesh.u

        rhs_nonlinear = np.zeros(self.mesh.N_nodes)
        # loop over elements
        for elem_idx in range(self.mesh.N_elems):
            # global node indices of the element
            node_idx = self.mesh.element_connectivity[elem_idx]
            # compute the local nonlinear rhs contribution sum_j f(phi_j u_j) 
            # f = self.f_nonlinear_local(u)
            f = np.zeros(self.mesh.N_dofs)
            t = 0
            for k in range(self.quadrature.N_quad):
                factor = self.quadrature.weights[k] * self.Jacobian_det
                # compute rhs contribution rhs_loc[i] =  f * phi_i
                for i in range(self.mesh.N_dofs):
                    for j in range(self.mesh.N_dofs):
                        t += self.basis.basis_vals[k,j]*u[j]
                    # f[k] = t(1-t)(1-2*t)
                    f[i] += t*self.basis.basis_vals[k,i]*factor
        return f
    
    def f_nonlinear_local(self, u):
        """
        Compute the nonlinear part of the rhs for one local element
        f(u) = u(1-u)(1-2u)
        sum f(phi_j*u_j)
        """
        f = np.zeros(self.quadrature.N_quad)
        for k in range(self.quadrature.N_quad):
            t = 0
            for j in range(self.mesh.N_dofs):
                t += self.basis.basis_vals[k,j]*u[j]
            f[k] =  t*(1-t)*(1-2*t)
            
        return f
    
    def plot_nonlinear_term(self, use_analytical = True, time_step = 0, output_dir = None):
        """
        Plot the nonlinear term spatial
        f(u(x1,x2))
        """
        global_min = np.min(self.mesh.u)
        global_max = np.max(self.mesh.u)

        # first run to get min max values
        f = np.zeros((self.mesh.N_nodes))
        for elem_idx in range(self.mesh.N_elems):
            u = self.mesh.u[self.mesh.element_connectivity[elem_idx]]
            f[self.mesh.element_connectivity[elem_idx]] = self.f_nonlinear_local(u)
        global_min = np.min(f)
        global_max = np.max(f)


        fig, ax1 = plt.subplots(1,1, figsize = (10,10))
        for elem_idx in range(self.mesh.N_elems):
            x_coords = self.mesh.coord[self.mesh.element_connectivity[elem_idx],0]
            y_coords = self.mesh.coord[self.mesh.element_connectivity[elem_idx],1]
            if use_analytical:
                u = np.array(self.mesh.u_analytical[self.mesh.element_connectivity[elem_idx]])
            else:
                u = self.mesh.u[self.mesh.element_connectivity[elem_idx]]
            u = np.array(u)
            f = self.f_nonlinear_local(u)
            #reshape
            x = x_coords.reshape((self.mesh.dof+1, self.mesh.dof+1))
            y = y_coords.reshape((self.mesh.dof+1, self.mesh.dof+1))
            f = f.reshape((self.mesh.dof+1, self.mesh.dof+1))
            mesh = ax1.pcolormesh(x, y, f, cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)

        ax1.set_title("Nonlinear term f(u(x1,x2)), at time step " + str(time_step))
        fig.colorbar(mesh, ax = ax1)

        if output_dir is None:
            output_dir = "/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/"

        output_dir += "/non_linear_term/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fname = f"{output_dir}/time{time_step}.png"

        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Nonlinear term plot saved at", fname)





    def cg(self, x0, rhs, tol = 1e-6, max_iter = 10000):
        """
        Conjugate Gradient method
        """
        x = x0
        r = rhs - self.compute_Au(x)
        r0 = np.linalg.norm(r)
        p = r
        rsold = np.dot(r,r)
        # time measurement
        for i in range(max_iter):
            Ap = self.compute_Au(p)
            alpha = rsold / np.dot(p,Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r,r)
            if np.sqrt(rsnew) < tol * r0:
                return x

            p = r + (rsnew/rsold) * p
            rsold = rsnew

        print("Did not converge after", max_iter, "iterations", "Residual norm:", np.sqrt(rsnew))
        return x

    def compute_Au(self, u):
        """ 
        Compute the matrix-vector product Au 
        for allen cahn it is 
        (u(n+1), nu) + Kdt a(u(n+1), nu) = (u(n), nu) - Kdt F(u(n), nu)
        just do mass matrix multiplied in each element and then multiply by bilinear dg form still as global matrix

        the rhs part is computed in the compute_rhs function
        """
        Au = np.zeros((self.mesh.N_nodes))


        if True:
            factor = 1
            M_loc = self.compute_local_mass_matrix()
            for elem_idx in range(self.mesh.N_elems):
                idx = self.mesh.element_connectivity[elem_idx]
                Au[idx] += factor * np.dot(M_loc,u[idx])

            Au += self.mesh.K_ac * self.mesh.dt * np.dot(self.A,u)
            return Au
        else:
            # loop over elements to compute volume contribution
            for elem_idx in range(self.mesh.N_elems):
                idx = self.mesh.element_connectivity[elem_idx]
                S_loc = self.compute_local_stiffness_matrix(elem_idx)
                Au[idx] += np.dot(S_loc,u[idx])


            # loop over interior faces to compute face integral
            for face_idx in range(self.mesh.N_faces):
                if not(self.mesh.boundary_faces[face_idx]):
                    # get the global indices of the nodes to multiply the right values
                    elem_idx_E1, elem_idx_E2 = self.mesh.face_neighbours[face_idx]

                    # get the global indices of the nodes
                    idx_E1 = self.mesh.element_connectivity[elem_idx_E1]
                    idx_E2 = self.mesh.element_connectivity[elem_idx_E2]

                    M11, M12, M21, M22 = self.compute_local_interior_face_integral(face_idx)
                    Au[idx_E1] += np.dot(M11,u[idx_E1]) + np.dot(M12,u[idx_E2])
                    Au[idx_E2] += np.dot(M21,u[idx_E1]) + np.dot(M22,u[idx_E2])


            # boundary_side/face 0 and 2 have global indices at position 1
            # boundary_side/face 1 and 3 have global indices at position 0
            node_position = [1,0,1,0]
            
            # loop over boundary faces to compute face integral
            for boundary_side in range(4):
                # exterior
                if self.mesh.is_exterior_boundary[boundary_side]:
                    # loop over faces
                    for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                        # set the right global node indices for phase because others are -1
                        # get the element index
                        elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                        # get the global node indices of the element
                        idx_E1 = self.mesh.element_connectivity[elem_idx]

                        # face integral
                        M11, _ = self.compute_local_exterior_boundary_face_integral(face_idx, boundary_side)

                        Au[idx_E1] += np.dot(M11,u[idx_E1])


                # interior
                else:
                    for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                        # set the right global node indices for phase because others are -1
                        # get the element index
                        elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                        # get the global node indices of the element
                        idx_E1 = self.mesh.element_connectivity[elem_idx]
                        # face integral
                        M11, _ = self.compute_local_interior_boundary_face_integral(face_idx)
                        
                        Au[idx_E1] += np.dot(M11,u[idx_E1])
            return Au
        
    def set_linear_operator(self):
        """
        Set a scipy linear operator to calculate the solution of the system with the help of gmres

        """
        A = LinearOperator((self.mesh.N_nodes,self.mesh.N_nodes), matvec = self.compute_Au)
        self.A_linear_operator = A


    def build_global_matrix(self):
        """ 
        Build global matrix to solve the poisson problem
        """
        # initialize the global matrix
        A = np.zeros((self.mesh.N_nodes,self.mesh.N_nodes))
        rhs_loc = np.zeros((self.mesh.N_nodes))
        # loop over elements to compute volume contribution
        for elem_idx in range(self.mesh.N_elems):
            idx = self.mesh.element_connectivity[elem_idx]
            S = self.compute_local_stiffness_matrix(elem_idx)
            A[np.ix_(idx,idx)] +=  S

        # loop over interior faces to compute face integral
        for face_idx in range(self.mesh.N_faces):
            if not(self.mesh.boundary_faces[face_idx]):
                # get the global indices of the nodes to multiply the right values
                elem_idx_E1, elem_idx_E2 = self.mesh.face_neighbours[face_idx]

                # get the global indices of the nodes
                idx_E1 = self.mesh.element_connectivity[elem_idx_E1]
                idx_E2 = self.mesh.element_connectivity[elem_idx_E2]

                M11, M12, M21, M22 = self.compute_local_interior_face(face_idx)
                A[np.ix_(idx_E1,idx_E1)] += M11
                A[np.ix_(idx_E1,idx_E2)] += M12
                A[np.ix_(idx_E2,idx_E1)] += M21
                A[np.ix_(idx_E2,idx_E2)] += M22
                
        # boundary_side/face 0 and 2 have global indices at position 1
        # boundary_side/face 1 and 3 have global indices at position 0
        node_position = [1,0,1,0]
        # loop over boundary faces to compute face integral
        for boundary_side in range(4):
            # exterior
            if self.mesh.is_exterior_boundary[boundary_side]:
                # loop over faces
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # set the right global node indices for phase because others are -1
                    # get the element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the global node indices of the element
                    idx_E1 = self.mesh.element_connectivity[elem_idx]

                    # face integral
                    M11 = self.compute_local_exterior_boundary_face(face_idx, boundary_side)

                    A[np.ix_(idx_E1,idx_E1)] +=  M11 #alpha is either 1 or zero for dirichlet or neumann
            # interior
            else:
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # set the right global node indices for phase because others are -1
                    # get the element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the global node indices of the element
                    idx_E1 = self.mesh.element_connectivity[elem_idx]
                    # face integral
                    M11 = self.compute_local_interior_boundary_face(face_idx, boundary_side)
                    

                    A[np.ix_(idx_E1,idx_E1)] +=  M11
        return A

    def solve_allen_cahn(self):
        """ 
        Solve the Allen-Cahn problem and store results in a file.
        """
        self.mesh.set_rhs(self.compute_rhs())

        # Create unique output directory based on timestamp
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        output_dir = f"/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/{time}"
        
        print("Initial solution")
        radius_analytical_0, radius_numerical_0, time_step_0 = self.mesh.plot_solution1D(y=0.0, time_step=0, output_dir=output_dir)
        self.mesh.plot_solution2D(time_step=0)
        # File to store radius data
        radius_file = f"{output_dir}/radius_data.txt"
        
        # Write metadata and initial values
        write_radius_data(radius_file, metadata={
            "time_resolution": self.mesh.dt, 
            "mesh_size": self.mesh.cell_shape,
            "output_directory": output_dir
        }, values=[(time_step_0, radius_analytical_0, radius_numerical_0)], mode="w")

        modulo_plot = 1
        eval_count = 0
        self.plot_nonlinear_term(use_analytical = True, time_step = 0, output_dir = output_dir)
        # Time stepping loop
        for i in range(1, self.mesh.N_timestep + 1):
            counter = gmres_counter()
            u_old = self.mesh.u.copy()
            self.mesh.set_rhs(self.compute_rhs())

            u_new, output_code = gmres(self.A_linear_operator, self.mesh.rhs, x0=u_old, 
                                    rtol=self.linear_solver_tol, restart=1000, maxiter=1000, callback=counter)
            self.mesh.set_u(u_new)
            self.plot_nonlinear_term(use_analytical = False, time_step = i, output_dir = output_dir)

            if i % modulo_plot == 0:
                print("Timestep:", i)
                radius_analytical, radius_numerical, time_step = self.mesh.plot_solution1D(y=0.0, time_step=i, output_dir=output_dir)
                
                # Append to file
                write_radius_data(radius_file, values=[(time_step, radius_analytical, radius_numerical)], mode="a")
        
        print(f"Radius data saved at {radius_file}")
        return radius_file

    def compute_mass_matrix_product(self, u = None):
        """
        Compute the product of the mass matrix with u
        """
        if u is None:
            u = self.mesh.u
            
        M_loc = self.compute_local_mass_matrix()
        # loop over elements
        product = np.zeros(self.mesh.N_nodes)
        for elem_idx in range(self.mesh.N_elems):
            idx = self.mesh.element_connectivity[elem_idx]
            product[idx] = np.dot(M_loc,u[idx])
        return product
    
    def compute_rhs_dg(self, u = None):
        """
        Compute the rhs contribution of the dg discretisation of the laplacian
        this usually always takes none because it should always use the updated u
        not always the u from previous timestep
        """
        if u is None:
            u = self.mesh.u

        rhs_dg = np.zeros(self.mesh.N_nodes)
        node_position = [1,0,1,0]
        # rhs_volume = rhs.copy()
        # plt.plot(rhs, label = "rhs")
        # loop over boundary faces and check if theyre exterior or interior boundary
        for boundary_side in range (4):
            #exterior 
            if self.mesh.is_exterior_boundary[boundary_side]:
                # loop over boundary faces
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # get element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the node indices of the face
                    idx_E1 = self.mesh.element_connectivity[elem_idx]
                    # compute local rhs
                    rhs_dg[idx_E1] += self.compute_rhs_exterior_boundary_face_contribution(face_idx, boundary_side)
                # plt.plot(rhs, label = str(boundary_side))
                    # print("face", face_idx)
                    # print(self.compute_rhs_exterior_boundary_face_contribution(face_idx, boundary_side))
            #interior
            else:
                # print("boundary side:", boundary_side)
                for face_idx in self.mesh.boundary_faces_location[boundary_side]:
                    # get element index
                    elem_idx = self.mesh.face_neighbours[face_idx][node_position[boundary_side]]
                    # get the node indices of the face
                    idx_E1 = self.mesh.element_connectivity[elem_idx]
                    # compute local rhs
                    rhs_dg[idx_E1] +=  self.compute_rhs_interior_boundary_face(face_idx, boundary_side)
                    # print("face", face_idx)
                    # print(self.compute_rhs_interior_boundary_face(face_idx, boundary_side))
                # plt.plot(rhs, label = str(boundary_side))
        return rhs_dg
                




    def update_rhs(self):
        """
        Update the rhs of the system inside schwartz iterations 
        rhs = nl + mass + K*dt * dg
        nl and mass use u_old while dg uses u
        """
        rhs_nl = self.compute_nonlinear_rhs(self.mesh.u_old)
        rhs_mass = self.compute_mass_matrix_product(self.mesh.u_old)
        rhs_dg = self.compute_rhs_dg()
        rhs = rhs_nl + rhs_mass + self.mesh.K_ac * self.mesh.dt * rhs_dg
        self.mesh.set_rhs(rhs)
        return rhs


    def allen_cahn_time_step(self):
        """
        Solve the allen cahn equation in one time step
        """
        counter = gmres_counter()
        u_start = self.mesh.u.copy()
        u_new, output_code = gmres(self.A_linear_operator, self.mesh.rhs, x0=u_start, 
                                    rtol=self.linear_solver_tol, restart=1000, maxiter=1000, callback=counter)
        if output_code > 0:
            print("output code", output_code)

        self.mesh.set_u(u_new)
        

        return self.mesh.u


    def solve_poisson_matrix_free(self):
        """ solve the poisson problem matrix free"""
        self.mesh.set_rhs(self.compute_rhs())
        rhs = self.mesh.rhs
        u0 = self.mesh.u.copy()
        # u0 = np.zeros((self.mesh.N_nodes))
        # self.mesh.set_u(self.cg(u0, rhs))

        counter = gmres_counter()

        u_h, output_code = gmres(self.A_linear_operator, self.mesh.rhs, x0 = u0, rtol = self.linear_solver_tol, restart = 1000, maxiter=1000, callback=counter)
        if output_code > 0:
            print("output code", output_code)

        self.mesh.set_u(u_h)
        return self.mesh.u

    def compute_error_with_overlap(self):
        """ 
        Compute the L2 error and H0 error on the full subdomain
        L2 = sqrt(e_h^T*M*e_h)
        H0 = e_h^T*S*e_h
        """
        # loop over the elements

        L2_error = 0
        H0_error = 0
        for elem_idx in range(self.mesh.N_elems):
            # element nodes
            elem_node_indices = self.mesh.element_connectivity[elem_idx]
            M = self.compute_local_mass_matrix()
            S = self.compute_local_stiffness_matrix(elem_idx)
            L2_error += np.dot(self.mesh.e_h[elem_node_indices], np.dot(M, self.mesh.e_h[elem_node_indices]))
            H0_error += np.dot(self.mesh.e_h[elem_node_indices], np.dot(S, self.mesh.e_h[elem_node_indices]))
        
        self.mesh.L2_error_with_overlap = np.sqrt(L2_error)
        self.mesh.H0_error_with_overlap = np.sqrt(H0_error)
        return L2_error, H0_error

    def get_non_overlapping_elements(self):
        """ 
        Get the indices of the non-overlapping elements
        """
        # get array of all non overlapping elements by boolean invert
        overlapping_sides = np.invert(self.mesh.is_exterior_boundary)
        element_indices = np.arange(self.mesh.N_elems).reshape([self.mesh.cell_shape[1],self.mesh.cell_shape[0]])

        # check if mesh is no submesh
        if not(self.mesh.parent_mesh is None):
            # remove row or column at the sides where overlapping sides is true
            if self.mesh.overlap > 0:
                if overlapping_sides[0]:
                    element_indices = np.delete(element_indices,np.s_[:self.mesh.overlap],axis = 1)
                if overlapping_sides[1]:
                    element_indices = np.delete(element_indices,np.s_[-self.mesh.overlap:],axis = 1)
                if overlapping_sides[2]:
                    element_indices = np.delete(element_indices,np.s_[:self.mesh.overlap],axis = 0)
                if overlapping_sides[3]:
                    element_indices = np.delete(element_indices,np.s_[-self.mesh.overlap:],axis = 0)

        # flatten the array
        element_indices = element_indices.flatten()

        return element_indices

    def compute_error_without_overlap(self, use_raw = False):
        """   
        Compute the L2 error and H0 error on the non-overlapping part of the subdomain
        """
        # get the non overlapping elements
        element_indices = self.get_non_overlapping_elements()

        # init errors
        L2_error = 0
        H0_error = 0
        L2_error_np_norm = 0 

        M = self.compute_local_mass_matrix()

        # loop over the elements
        for elem_idx in element_indices:
            elem_node_indices = self.mesh.element_connectivity[elem_idx]
            S = self.compute_local_stiffness_matrix(elem_idx)
            # M = self.compute_local_mass_matrix()
            

            if use_raw:
                e_h = self.mesh.e_h_raw[elem_node_indices]
            else:   
                e_h = self.mesh.e_h[elem_node_indices]

            L2_error += np.dot(e_h, np.dot(M, e_h))
            L2_error_np_norm += np.linalg.norm(e_h, ord = 2)**2
            H0_error += np.dot(e_h, np.dot(S, e_h))
       
        if use_raw:
            self.mesh.L2_error_without_overlap = L2_error
            self.mesh.H0_error_without_overlap = H0_error
        else:
            self.mesh.L2_error_without_overlap = np.sqrt(L2_error)
            self.mesh.H0_error_without_overlap = np.sqrt(H0_error)
        
        
        # print("diff of L2 errors", L2_error_np_norm - L2_error)
        return self.mesh.L2_error_without_overlap, self.mesh.H0_error_without_overlap
    


    def compute_residual(self):
        """ 
        Compute the residual of the DG problem 
        without overlap 
        """
        
        residual_vec = self.compute_residual_vector()
        residual_denominator = self.compute_denominator_residual()

        # Compute the residual using the residual vector
        residual = np.linalg.norm(residual_vec) / np.linalg.norm(residual_denominator)
        
        return residual

    def compute_residual_vector(self):
        """ 
        Compute residual vector of the DG problem
        """
        Au = self.compute_Au(self.mesh.u)
        
        # get the non overlapping elements
        # non_overlapping_elem_indices = self.get_non_overlapping_elements()
        # # get the global node indices of the non overlapping elements
        # global_idx = np.hstack([self.mesh.element_connectivity[elem_idx] for elem_idx in non_overlapping_elem_indices])

        # with overlapping elements
        global_idx = np.hstack([self.mesh.element_connectivity[elem_idx] for elem_idx in range(self.mesh.N_elems)])
        rhs_old = self.mesh.rhs
        # print("inside residual diff", np.linalg.norm(rhs_loc[global_idx]-rhs_old[global_idx]))
        # compute the residual vector
        residual_vector = (rhs_old[global_idx] - Au[global_idx])

        return residual_vector
    

    def compute_denominator_residual(self):
        """ 
        Compute the denominator of the residual for the DG problem to norm the residual vector
        """
        # get the non overlapping elements
        non_overlapping_elem_indices = self.get_non_overlapping_elements()
        # get the  node indices of the non overlapping elements
        local_idx = np.hstack([self.mesh.element_connectivity[elem_idx] for elem_idx in non_overlapping_elem_indices])
        
        # if the mesh is a submesh
        if not(self.mesh.parent_mesh is None):
            # map the local to the global indices
            global_elem_indices = self.mesh.mapping_subdomain2global_elements.flatten()
            global_idx_non_overlapping = np.hstack([self.mesh.parent_mesh.element_connectivity[global_elem_indices[elem_idx]] for elem_idx in non_overlapping_elem_indices])
            global_idx_overlapping = np.hstack([self.mesh.parent_mesh.element_connectivity[global_elem_indices[elem_idx]] for elem_idx in range(self.mesh.N_elems)])
            u0_local = self.mesh.parent_mesh.u0[global_idx_overlapping]

        else:
            global_elem_indices = np.arange(self.mesh.N_elems)
            global_idx_overlapping = np.hstack([self.mesh.element_connectivity[elem_idx] for elem_idx in range(self.mesh.N_elems)])
            u0_local = self.mesh.u0[global_idx_overlapping]
            
        # compute the local u0 and Au0
        Au0 = self.compute_Au(u0_local)


        # compute the denominator to norm the residual vector
        denominator_residual = self.mesh.rhs[local_idx] - Au0[local_idx]
        
        return denominator_residual


def write_radius_data(file_path, metadata=None, values=None, mode="w"):
    """
    Writes radius values and metadata to a file.
    :param file_path: Path to the file
    :param metadata: Dictionary with metadata (only written if mode="w")
    :param values: List of (time_step, radius_analytical, radius_numerical)
    :param mode: "w" for write (overwrites), "a" for append
    """
    with open(file_path, mode) as f:
        if metadata and mode == "w":
            f.write("# Metadata\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("# time_step, radius_analytical, radius_numerical\n")

        if values:
            for time_step, radius_analytical, radius_numerical in values:
                f.write(f"{time_step} {radius_analytical} {radius_numerical}\n")
