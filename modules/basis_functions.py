import numpy as np

from .quadrature import Quadrature


class BasisFunction:
    """ 
    Class for defining basis functions
    """
    def __init__(self,degree,quadrature):
        """ 
        Constructor of the BasisFunction class
        Input:
            degree: (int) degree of the basis functions
        Attributes:
            degree: (int) degree of the basis functions
            nodes_1D: (np.array) Lagrange nodes in 1D
            nodes_2D: (np.array) Lagrange nodes in 2D
            quadrature: (Quadrature) quadrature object
            basis_funcs: (np.array) basis functions evaluated at the quadrature points
        """
        self.degree = degree # degree of the basis functions 
        self.nodes_1D, self.nodes_2D = self.get_lagrange_nodes() # points where the basis functions are evaluated
        self.quadrature = quadrature # defining the quadrature object to evaluate the basis functions on the quadrature points
        self.basis_vals = self.compute_values_on_quadrature_points() # basis functions evaluated on the quadrature points
        self.basis_der_vals = self.compute_der_values_on_quadrature_points() # basis function derivatives evaluated on the quadrature points
        self.basis_vals_face_integral = self.compute_values_at_face_quadrature_points() # basis functions evaluated on the quadrature points of the face
        self.basis_der_vals_face_integral = self.compute_der_values_at_face_quadrature_points() # basis function derivatives evaluated on the quadrature points of the face
        
    def get_lagrange_nodes(self):
        """ 
        Generate Lagrange nodes on the reference element [0, 1] x [0, 1]
        Output:
            nodes_1D: (np.array) Lagrange nodes in 1D
            nodes_2D: (np.array) Lagrange nodes in 2D
        """
        points = np.linspace(0, 1, self.degree + 1) # equidistant points on reference element
        nodes_1D = points
        temp = np.array(np.meshgrid(points, points)).reshape(2, -1)
        nodes_2D = temp.T
        return nodes_1D, nodes_2D

    def basis_val_1D(self, xi, i):
        """ 
        Compute the Lagrangian basis function in 1D
        """
        L = 1
        for j, node in enumerate(self.nodes_1D):
            if j != i:
                L *= (xi - node) / (self.nodes_1D[i] - node)
        return L
    
    def basis_val_1D_derivative(self, xi, i):
        """ 
        Compute the derivative of the Lagrangian basis function in 1D
        """
        L_prime = 0
        for j, node_j in enumerate(self.nodes_1D):
            if j != i:
                term = 1 / (self.nodes_1D[i] - node_j)
                for k, node_k in enumerate(self.nodes_1D):
                    if k != i and k != j:
                        term *= (xi - node_k) / (self.nodes_1D[i] - node_k)
                L_prime += term
        return L_prime
    
    def compute_basis_functions_point(self,x):
        """ 
        Evaluate laplacian basis functions for a given point x
        Input:
            x: (np.array) point in the reference element 2D
        Output:
            basis_funcs: (np.array) basis functions evaluated at point x
        """
        basis_funcs = np.zeros(len(self.nodes_2D))                
        for i_y in range(self.degree+1):
            for i_x in range(self.degree+1):
                i = i_y*(self.degree+1) + i_x
                L_x = self.basis_val_1D(x[0], i_x)
                L_y = self.basis_val_1D(x[1], i_y)
                basis_funcs[i] = L_x * L_y
        return basis_funcs
    
    def compute_basis_function_point_derivative(self,x):
        """ 
        Compute the derivative of the basis functions at a given point x
        Input:
            x: (np.array) point in the reference element 2D
        Output:
            basis_func_derivatives: (np.array) basis function derivatives at point x
        """
        basis_func_derivatives = np.zeros((len(self.nodes_2D), 2))
        for i_y in range(self.degree+1):
            for i_x in range(self.degree+1):
                i = i_y*(self.degree+1) + i_x
                L_x = self.basis_val_1D(x[0], i_x)
                L_y = self.basis_val_1D(x[1], i_y)
                L_x_prime = self.basis_val_1D_derivative(x[0], i_x)
                L_y_prime = self.basis_val_1D_derivative(x[1], i_y)
                basis_func_derivatives[i, 0] = L_x_prime * L_y
                basis_func_derivatives[i, 1] = L_x * L_y_prime
        return basis_func_derivatives
    
    def compute_values_on_quadrature_points(self):
        """ 
        compute the basis functions on the quadrature points
        Output:
            values: (np.array) N_quad_2D x dof+1 basis functions evaluated at the quadrature points
        """
        values = np.zeros((self.quadrature.points.shape[0],self.nodes_2D.shape[0]))
        for i, point in enumerate(self.quadrature.points):
            values[i,:] = self.compute_basis_functions_point(point)
        return values
    
    def compute_der_values_on_quadrature_points(self):
        """
        compute the derivative of the basis functions on the quadrature points
        Output:
            values: (np.array) N_quad_2D x dof+1 x 2 basis function derivatives evaluated at the quadrature points
        """
        values = np.zeros((self.quadrature.points.shape[0], self.nodes_2D.shape[0],2))
        for i, point in enumerate(self.quadrature.points):
            values[i,:,:] = self.compute_basis_function_point_derivative(point)
        return values
    
    def compute_values_at_face_quadrature_points(self):
        """ 
        Compute the basis function values on the quadrature points of the face -> 1D integral
        Output:
            values: 4 x N_quad_1D x (degree+1)^2 (np.array) basis function values on the quadrature points for each face
            (ordering like always [left,right,bottom,top])
            could be changed to just degree+1 bc rest should be zero
        """
        values = np.zeros((4,self.quadrature.points_1D.shape[0],(self.degree+1)**2))
        for i, point in enumerate(self.quadrature.points_1D):
            values[0,i,:] = self.compute_basis_functions_point([0,point])
            values[1,i,:] = self.compute_basis_functions_point([1,point])
            values[2,i,:] = self.compute_basis_functions_point([point,0])
            values[3,i,:] = self.compute_basis_functions_point([point,1])
        return values
    
    def compute_der_values_at_face_quadrature_points(self):
        """ 
        Compute the basis function derivatives on the quadrature points of the face -> 1D integral
        Output:
            values: 4 x 2 x N_quad x (dof+1)^2 (np.array) basis function derivatives on the quadrature points for each face
            (ordering like always [left,right,bottom,top])
        """
        values = np.zeros((4,2,self.quadrature.points_1D.shape[0],(self.degree+1)**2))
        for i, point in enumerate(self.quadrature.points_1D):
            values[0,0,i,:] = self.compute_basis_function_point_derivative([0,point])[:,0]
            values[0,1,i,:] = self.compute_basis_function_point_derivative([0,point])[:,1]
            values[1,0,i,:] = self.compute_basis_function_point_derivative([1,point])[:,0]
            values[1,1,i,:] = self.compute_basis_function_point_derivative([1,point])[:,1]
            values[2,0,i,:] = self.compute_basis_function_point_derivative([point,0])[:,0]
            values[2,1,i,:] = self.compute_basis_function_point_derivative([point,0])[:,1]
            values[3,0,i,:] = self.compute_basis_function_point_derivative([point,1])[:,0]
            values[3,1,i,:] = self.compute_basis_function_point_derivative([point,1])[:,1]
        return values
    

if __name__ == "__main__":
    quadrature = Quadrature(2)
    basis_functions = BasisFunction(2,quadrature)
    print("successful BasisFunction run")