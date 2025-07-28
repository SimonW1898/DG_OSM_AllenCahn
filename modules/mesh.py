# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D

import pickle
import os

import pandas as pd


class Mesh():
    """ 
    Structure for the global mesh
    """
    def __init__(self, cell_shape = (4,4), dof = 1, x_domain = [0,1], y_domain = [0,1]):
        # for allen cahn time interval is (0,1)
        self.K_ac = 1
        self.R_0 = 0.25
        self.epsilon_ac = 0.04
        
        self.dt = 1e-4
        self.T = 3.125e-2
        self.N_timestep = int(self.T/self.dt)
        


        # for meshing
        self.parent_mesh = None
        self.cell_shape = cell_shape
        self.dof = dof
        self.N_dofs = (dof+1)**2
        self.N_nodes = (self.cell_shape[0]*self.cell_shape[1])*self.N_dofs
        self.N_nodes_1D_x = self.cell_shape[0]*(dof+1)
        self.N_nodes_1D_y = self.cell_shape[1]*(dof+1)
        self.N_elems = self.cell_shape[0]*self.cell_shape[1]
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.N_faces = self.cell_shape[1]*(self.cell_shape[0]+1) + self.cell_shape[0]*(self.cell_shape[1]+1)
        self.hx = (x_domain[1] - x_domain[0])/self.cell_shape[0]
        self.hy = (y_domain[1] - y_domain[0])/self.cell_shape[1]
        # # self.coord = self.create_coord()
        self.single_mapping = self.get_single_mapping()
        self.local_mapping = self.get_local_mapping()
        self.coord, self.element_connectivity, self.bottom_left_index, self.bottom_left_idx_single = self.create_coord_in_cell()
        self.local_faces = self.get_local_faces()
        self.boundary_cells = self.get_boundary_cells()
        self.interior_cells = self.get_interior_cells()
        self.element_neighbours = self.get_neighbours()
        self.element_faces = self.create_faces()
        self.face_neighbours, self.global_dof_idx_on_faces = self.get_face_neighbours()
        self.boundary_faces = self.get_boundary_faces()
        self.boundary_faces_location = self.get_boundary_location_faces()
        self.is_vertical_face = self.get_vertical_faces()
        self.normal_vectors = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        # self.u = np.zeros(self.N_nodes) #self.analytical_solution()
        self.u_analytical = self.initial_condition()

        # set seed for reproducibility
        # random values between min(analytical_solution) and max(analytical_solution)
        if True:
            self.u = self.initial_condition()
        else:    
            self.u = np.zeros(self.N_nodes) #self.analytical_solution()

        self.u0 = np.copy(self.u)

        self.set_e_h()


        # added to solve dg with this mesh
        # boolean array with length 4 and value true
        self.is_exterior_boundary = np.ones((4),dtype=bool)

        # multiplication stencil for putting together the solution
        self.stencils = self.create_stencils()

        # classify each element with a case 
        self.element_position = self.get_element_position()


        
    def compute_R(self, n):
        """
        Compute the radius after n*dt time steps
        Computation is for mu = 0
        R = sqrt(R_0^2 -2K(t-t_0))
        """
        return np.sqrt(self.R_0**2 - 2*self.K_ac*n*self.dt)

    def set_u(self, u):
        self.u = u
        self.set_e_h()
            
    def set_e_h(self):
        # self.e_h = (self.u - self.u_analytical)/np.max(np.abs(self.u_analytical))
        self.e_h = (self.u - self.u_analytical)/np.linalg.norm(self.u_analytical)
        self.e_h_raw = self.u - self.u_analytical
        

    def set_rhs(self, rhs):
        self.rhs = rhs

    def set_u_old(self, u = None):
        if u is None:
            self.u_old = np.copy(self.u)
        else:
            self.u_old = u

    def get_single_mapping(self):
        """ 
        for coordinates with just single nodes
        get  mapping such that just the bottom left index needs of a cell needs
        to be added to global mapping to get the global indices of an cell (respecting the lexicographical ordering)
        Output:
            - array (dof+1)^2 containing the global mapping
        """
        mapping = np.zeros((self.dof+1)**2,dtype=int)
        start = 0
        for i in range(self.dof+1):
            for j in range(self.dof+1):
                idx = i*(self.dof+1) + j
                mapping[idx] = start +  j
            start = start + self.cell_shape[0]*self.dof + 1 # plus x because of lexico ordering 
        return mapping
    
    def get_local_mapping(self):
        """ 
        get local mapping such that just the bottom left index needs to be added to mapping to get the global indices of an element with doubling
            Output:
                - array (dof+1)^2 containing the local mapping
        """
        mapping = np.zeros((self.dof+1)**2,dtype=int)
        start = 0
        for i in range(self.dof+1):
            for j in range(self.dof+1):
                idx = i*(self.dof+1) + j
                mapping[idx] = start +  j
            start = start + self.N_nodes_1D_x
        return mapping
    
    def create_coord_in_cell(self):
        """ 
        Create a matrix containing the coordinates of the nodes of the cell
        Output:
            - coord_double: N_dofs x 2 coordinates of the nodes of the cell with doubling
            - element_connectivity: N_elems x N_dofs matrix containing the indices of the nodes of each element
            - bottom_left_idx_doubling: N_elems x 1 matrix containing the bottom left index of the element

        """
        # create the bottom left index of the all cells
        bottom_left_idx_doubling = np.zeros((self.N_elems),dtype=int)
        bottom_left_idx_single = np.zeros((self.N_elems),dtype=int)
        
        bottom_left_doubling_add = 0
        bottom_left_single_add = 0

        y_direction_doubling_addition =  self.cell_shape[0]*(self.N_dofs)
        y_direction_single_addition =    self.cell_shape[0]*(self.dof**2) + self.dof

        bottom_left_idx_doubling[0] = bottom_left_doubling_add
        bottom_left_idx_single[0] = bottom_left_single_add

        for i in range(self.cell_shape[1]):
            for j in range(self.cell_shape[0]):
                if i == 0 and j == 0:
                    continue
                idx = i*self.cell_shape[0] + j

                bottom_left_idx_doubling[idx] = j*(self.dof + 1) + bottom_left_doubling_add
                bottom_left_idx_single[idx] = j*self.dof + bottom_left_single_add

            bottom_left_doubling_add += y_direction_doubling_addition
            bottom_left_single_add += y_direction_single_addition


        # create the coordinates of the nodes in the cell 
        # initalialize coordinates for the cell with doubling
        coord_double = np.zeros((self.N_elems * self.N_dofs,2))
        # with single nodes
        x = np.linspace(self.x_domain[0], self.x_domain[1], (self.dof)*self.cell_shape[0]+1)
        y = np.linspace(self.y_domain[0], self.y_domain[1], (self.dof)*self.cell_shape[1]+1)

        xv, yv = np.meshgrid(x, y)  # meshgrid for the nodes
        coord_single = np.vstack([xv.ravel(), yv.ravel()]).T

        # initialize element connecitvity matrix
        element_connectivity = np.zeros((self.N_elems,self.N_dofs),dtype=int)
        # loop over the cells
        for i in range(self.N_elems):
            coord_double[bottom_left_idx_doubling[i]+self.local_mapping] = coord_single[bottom_left_idx_single[i]+self.single_mapping]
            element_connectivity[i,:] = bottom_left_idx_doubling[i] + self.local_mapping
        return coord_double, element_connectivity, bottom_left_idx_doubling, bottom_left_idx_single

    def get_local_faces(self):
        """ 
        Create a matrix containing the local indices of the faces of one cell
        Output:
            4 x N_dof+1 (int) matrix containing the indices of the nodes of each face
        """
        local_faces = np.zeros((4,self.dof+1),dtype=int)
        for i in range(self.dof+1):
            local_faces[0,i] = i*(self.dof+1)
            local_faces[1,i] = (i)*(self.dof+1)+self.dof
            local_faces[2,i] = i
            local_faces[3,i] = (self.dof+1)*self.dof + i
        return local_faces

    def get_face_indices(self, elem_idx):
        """ 
        return the matrix containing the global face indices of the element
        Output:
            4 x N_dof+1 (int) matrix containing the indices of the nodes of each face
        """
        global_faces = np.zeros((4,self.dof+1),dtype=int)
        global_faces = self.element_connectivity[elem_idx,self.local_faces]
        
        return global_faces



    def create_faces(self):
        """ 
        Create a Matrix containing all node indices of the face on the global mesh
        Output:
            output:
                - cells: N_elems x N_dofs (int) matrix containing the indices of the nodes of each face
                - faces: N_elems x 4 x dof+1  (int) List that contains all face indices for each cell
                - boundaries: [cell_shape[1], cell_shape[1], cell_shape[0], cell_shape[0]] list of (int) List that contains the indices of the boundary cells
                - local_face_indexing: 4 x N_elem_1D (np.array) 4 list containing elements indices of the boundary elements, just add this to bottom left index to get the global indexing
        """
        cells = np.zeros((self.N_elems, self.N_dofs),dtype=int)
        faces = np.zeros((self.N_elems,4,self.dof+1),dtype=int)

        boundaries = [np.zeros(self.cell_shape[1],dtype=int),np.zeros(self.cell_shape[1],dtype=int),\
                      np.zeros(self.cell_shape[0],dtype=int),np.zeros(self.cell_shape[0],dtype=int)]

        local_indexing = np.zeros((4,self.N_dofs),dtype=int)
    

    def get_boundary_cells(self):
        """ 
        Create List containing the indices of the boundary cells
        Output:
            [[cell_shape[1]],[cell_shape[1]],[cell_shape[0]],[cell_shape[0]]] list of (int) List that contains the indices of the boundary cells
        """
                # Generate boundary cell lists using vectorized operations
        bottom_boundary = np.arange(self.cell_shape[1]) * self.cell_shape[0]
        top_boundary = bottom_boundary + self.cell_shape[0] - 1
        left_boundary = np.arange(self.cell_shape[0])
        right_boundary = left_boundary + self.cell_shape[0] * (self.cell_shape[1] - 1)

        # Combine the boundary lists into the final list
        boundary_cell_list = [bottom_boundary, top_boundary, left_boundary, right_boundary]

        return boundary_cell_list

    def get_interior_cells(self):
        """  
        Array of element indices that are interior cells (not on the boundary)
        """
        # Generate all element indices
        all_elements = np.arange(self.N_elems)

        # Cast the boundary elements to a 1D array
        boundary_elements = np.concatenate(self.boundary_cells)

        # Get the interior elements
        interior_elements = np.setdiff1d(all_elements, boundary_elements)

        return interior_elements

    def get_neighbours(self):
        """ 
        Get the neighbours of the global mesh
        Output:
            4 x N_cells numpy array containing the indices of the neighbours of the cells
            [left,right,bottom,top] -1 if its a boundary cell on that side
        """

        # Create the neighbour_stencil array
        neighbour_stencil = np.array([-1, 1, -self.cell_shape[0], self.cell_shape[0]])

        base_indices = np.arange(self.N_elems).reshape(-1, 1)

        neighbour_indices = base_indices + neighbour_stencil

        # mark boundary neighbours as -1
        # loop over boundary_cells
        for boundary_cell_idx, position in enumerate(self.boundary_cells):
            neighbour_indices[position,boundary_cell_idx] = -1

        return neighbour_indices    

    def create_faces(self):
        """ 
        Create a Matrix containing the face indices for each element
        Output:
            N_elems x 4 (int) matrix containing the indices of the nodes of each face
        """
        faces = np.zeros((self.N_elems,4),dtype=int)
        
        face_stencil = np.array([0,1,self.cell_shape[0]+1,2*(self.cell_shape[0]+1)+self.cell_shape[0]])
        base_indices = np.zeros((self.N_elems),dtype=int)
        
        # Calculate base indices with vectorisation
        rows = np.arange(self.cell_shape[1])
        cols = np.arange(self.cell_shape[0])

        base_indices = (rows[:, None] * (2 * self.cell_shape[0] + 1) + cols).flatten()

        # Add stencil to base indices
        faces = base_indices[:, None] + face_stencil
        # correct the faces for the last row because no vertical faces are there to be numbered before
        faces[-self.cell_shape[0]:,3] -= self.cell_shape[0] + 1
        return faces

    def get_face_neighbours(self):
        """ 
        Get the neighbouring elements to a face
        Get the global node indices on thtat face for each element
        Output:
            - neighbours: N_faces x 2 (int) matrix containing the indices of the neighbouring elements
            - global_dof_idx_on_faces: N_faces x 2 x dof+1 (int) matrix containing the indices of the nodes of each face
        """
        neighbours = np.zeros((self.N_faces,2),dtype=int)
        global_dof_idx_on_faces = -1*np.ones((self.N_faces,2,self.dof+1),dtype=int)
        

        left_bottom_face = np.array([0,2],dtype=int)
        right_top_face = np.array([1,3],dtype=int)

        # loop over elements
        for i in range(self.N_elems):
            global_face_indices = self.get_face_indices(i)
            element_neighbours = self.element_neighbours[i]
            element_faces = self.element_faces[i]
            #bottom and left are first entries in the arrays
            neighbours[element_faces[left_bottom_face],0] = element_neighbours[left_bottom_face]
            #top and right are second entries in the arrays
            neighbours[element_faces[right_top_face],1] = element_neighbours[right_top_face]

            # get the global node indices on the faces
            # bottom and left
            global_dof_idx_on_faces[element_faces[left_bottom_face],1,:] = global_face_indices[left_bottom_face,:]
            # top and right
            global_dof_idx_on_faces[element_faces[right_top_face],0,:] = global_face_indices[right_top_face,:]

            # handle bottom and left elements
            boundary_elements = self.boundary_cells
            # bottom cells
            neighbours[self.element_faces[boundary_elements[2],2],1] = boundary_elements[2]
            # left cells
            neighbours[self.element_faces[boundary_elements[0],0],1] = boundary_elements[0]
            # top cells
            neighbours[self.element_faces[boundary_elements[3],3],0] = boundary_elements[3]
            # right cells
            neighbours[self.element_faces[boundary_elements[1],1],0] = boundary_elements[1]
            
            
        return neighbours, global_dof_idx_on_faces

    def get_boundary_faces(self):
        """ 
        create a boolean array with true for a boundary face
        Output:
            - array N_faces x 1 containing boolean values
        """
        boundary_faces = np.zeros((self.N_faces),dtype=bool)

        for i in range(4):
            boundary_faces[self.element_faces[self.boundary_cells[i],i]] = True

        return boundary_faces
    
    def get_boundary_location_faces(self):
        """ 
        create a matrix containing the face indices of the boundary faces for each side
        Output:
            4 x cell_shape[0] (int) matrix containing the indices of the boundary faces
        """
        boundary_faces_location = [np.zeros_like(arr) for arr in self.boundary_cells]
        # left boundary
        for i in range(4):
            boundary_faces_location[i] = self.element_faces[self.boundary_cells[i],i]
        return boundary_faces_location
    
    def get_vertical_faces(self):
        """ 
        create a boolean array with true for a vertical face
        Output:
            - array N_faces x 1 containing boolean values
        """
        is_vertical_face = np.zeros((self.N_faces),dtype=bool)
        
        # loop over elements
        for i in range(self.N_elems):
            # get the face indices
            face_indices = self.element_faces[i]
            # set the right face to true for each element
            is_vertical_face[face_indices[1]] = True
        
        # loop over left boundary cells
        for idx in self.boundary_cells[0]:
            # get the face indices
            face_indices = self.element_faces[idx]
            # set the left face to true for each element
            is_vertical_face[face_indices[0]] = True
        return is_vertical_face


    def analytical_solution(self, x = None):
        """
        Compute analytical function value for a given point x inside the domain
        """
        if x is None:
            x = self.coord

        r = np.sqrt((x[:,0]**2 + x[:,1]**2))
        return 0.5 * (1 + np.tanh((self.R_0 - r) / (2 * self.epsilon_ac)))

    def initial_condition(self, x = None):
        if x is None:
            x = self.coord

        r = np.sqrt((x[:,0]**2 + x[:,1]**2))
        return 0.5 * (1 + np.tanh((self.R_0 - r) / (2 * self.epsilon_ac)))
    
    def grad_analytical_solution(self, x=None):
        """
        Compute the gradient of the analytical solution at a given point x.
        Returns the gradient as a vector [d/dx1, d/dx2].
        """
        if x is None:
            x = self.coord

        # Unpacking the coordinates
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Terms of the solution
        term1 = x1 * (x1 - 1)
        term2 = x2 * (x2 - 1)
        term3 = np.exp(-x1**2 - x2**2)

        # Gradients of the terms
        d_term1_dx1 = (2 * x1 - 1)
        d_term2_dx2 = (2 * x2 - 1)

        d_term3_dx1 = term3 * (-2 * x1)
        d_term3_dx2 = term3 * (-2 * x2)

        # Gradient of the analytical solution
        grad_x1 = (d_term1_dx1 * term2 * term3) + (term1 * term2 * d_term3_dx1)
        grad_x2 = (term1 * d_term2_dx2 * term3) + (term1 * term2 * d_term3_dx2)
        

        # neumann bc is zero for ac
        grad_x1 = 0*grad_x1
        grad_x2 = 0*grad_x2
        # Stack the gradients for each point into a 2D array
        grad = np.column_stack([grad_x1, grad_x2])

        
        return grad
    
    def create_stencils(self):
        """   
        make the stencils for all cases
        0: left boundary
        1: right boundary
        2: bottom boundary
        3: top boundary
        4:bottom left corner
        5: bottom right corner
        6: top left corner
        7: top right corner
        8: interior
        """
        stencils = np.ones((9,self.N_dofs))
        local_faces = self.get_local_faces()
        corner_indices = np.array([0,(self.dof+1)-1,self.N_dofs-(self.dof+1), self.N_dofs-1])
        # interior face nodes
        interior_face_nodes = local_faces[:,1:-1]

        for i in range(9):
            if i == 0:
                # left boundary
                stencils[i,interior_face_nodes[[1,2,3],:]] = 1/2
                # bottom right, top right corners have two nodes in interior
                stencils[i,corner_indices[[0,2]]] = 1/2
                # bottom + top right corner have 4 nodes in interior
                stencils[i,corner_indices[[1,3]]] = 1/4
            elif i == 1:
                # right boundary
                stencils[i,interior_face_nodes[[0,2,3],:]] = 1/2
                # bottom, top left corners have two nodes in interior
                stencils[i,corner_indices[[1,3]]] = 1/2
                # bottom + top left corner have 4 nodes in interior
                stencils[i,corner_indices[[0,2]]] = 1/4
            elif i == 2:
                # bottom boundary
                stencils[i,interior_face_nodes[[0,1,3],:]] = 1/2
                # top right + bottom right corner have two nodes in interior
                stencils[i,corner_indices[[0,1]]] = 1/2
                # top right, top left corners have 4 nodes in interior
                stencils[i,corner_indices[[2,3]]] = 1/4
            elif i == 3:
                # top boundary
                stencils[i,interior_face_nodes[[0,1,2],:]] = 1/2
                # bottom left + top left corner have two nodes in interior
                stencils[i,corner_indices[[2,3]]] = 1/2
                # bottom left, bottom right corners have 4 nodes in interior
                stencils[i,corner_indices[[0,1]]] = 1/4
            elif i == 4:
                # bottom and left have only one node in interior
                stencils[i,interior_face_nodes[[1,3],:]] = 1/2
                # bottom right, top left corner have two nodes in interior
                stencils[i,corner_indices[[1,2]]] = 1/2
                # top right corner has 4 nodes in interior
                stencils[i,corner_indices[3]] = 1/4
            elif i == 5:
                # bottom and right have only one node in interior
                stencils[i,interior_face_nodes[[0,3],:]] = 1/2
                # bottom left, top right corners have two nodes in interior
                stencils[i,corner_indices[[0,3]]] = 1/2
                # top left corner has 4 nodes in interior
                stencils[i,corner_indices[2]] = 1/4
            elif i == 6:
                # top and left have only one node in interior
                stencils[i,interior_face_nodes[[1,2],:]] = 1/2
                # bottom right, top left corners have two nodes in interior
                stencils[i,corner_indices[[0,3]]] = 1/2
                # bottom right corner has 4 nodes in interior
                stencils[i,corner_indices[1]] = 1/4
            elif i == 7:
                # top and right have only one node in interior
                stencils[i,interior_face_nodes[[0,2],:]] = 1/2
                # bottom left, top right corners have two nodes in interior
                stencils[i,corner_indices[[1,2]]] = 1/2
                # bottom left corner has 4 nodes in interior
                stencils[i,corner_indices[0]] = 1/4
            
            elif i == 8:
                # all interior face nodes have 2 nodes in interior
                stencils[i,interior_face_nodes] = 1/2
                # all corners have 4 nodes in interior
                stencils[i,corner_indices] = 1/4
        return stencils


    def get_element_position(self):
        """   
        Create array that classifies the element position from 0 to 8 see above
        """
        # init with 8 for interior elements
        element_position = np.ones((self.N_elems),dtype=int)*8
        # corner indices         
        corner_indices = np.array([0,self.cell_shape[0] - 1,self.N_elems-self.cell_shape[0], self.N_elems-1])
        # loop over boundary cells
        for i in range(4):
            # set the position to the index
            element_position[self.boundary_cells[i]] = i
        
        # loop over corner indices
        for i in range(4):
            element_position[corner_indices[i]] = i + 4
            
        return element_position

    def plot_coord(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot(self.coord[:, 0], self.coord[:, 1], 'bo')
        
        # Generate systematic offsets
        offset_grid = np.array([[0, 0], [0.05, 0.05], [-0.05, 0.05], [0.05, -0.05], [-0.05, -0.05]])
        num_offsets = offset_grid.shape[0]
        
        # Plot the numbering with a systematic offset to make them visible
        for i, (x, y) in enumerate(self.coord):
            offset = offset_grid[i % num_offsets]
            ax.text(x + offset[0], y+ offset[1], str(i), color='red', fontsize=8, ha='right')
            
        ax.set_title('Mesh')
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.show()

    def plot_mesh(self, ax = None, color = 'black', line_width=1.0):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')

        faces = [1,3,0,2]
        for i in range(self.N_elems):
            idx_faces = self.get_face_indices(i)
            for i in range(4):
                coord = self.coord[idx_faces[i]]
                polygon = patches.Polygon(coord, edgecolor=color, facecolor='none',linewidth=line_width)
                ax.add_patch(polygon)

            
        ax.plot(self.coord[:, 0], self.coord[:, 1], 'bo')
        
        ax.set_title('Mesh')
        plt.xlabel('X')
        plt.ylabel('Y')
        # if ax is None:
            #plt.show()

    def plot_solution(self, time_step=0, output_dir = None, radius = None):
        """
        Plot the solution with improved visualization for thesis use.
        Ensures consistent axis limits and color scaling.
        """
        # Define fixed axis limits
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
        z_min, z_max = 0, 1
        color_min, color_max = 0, 1  # Fix color scale to always show full range

        # Set up the figure with higher DPI for better quality
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Set perspective for better visualization
        ax.view_init(elev=30, azim=135)  # Adjust angle for clarity

        # Loop over elements
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof+1, self.dof+1))
            y = np.array(y_coords).reshape((self.dof+1, self.dof+1))
            values = np.array(self.u[self.element_connectivity[elem_idx]]).reshape((self.dof+1, self.dof+1))

            # Plot each element with fixed color scaling
            surface = ax.plot_surface(x, y, values, cmap='viridis', vmin=color_min, vmax=color_max, edgecolor='k', linewidth=0.3, antialiased=True)

        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Add color bar with fixed range
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Ensure 0 and 1 are always visible

        # Set axis labels
        ax.set_xlabel("X", fontsize=12, labelpad=10)
        ax.set_ylabel("Y", fontsize=12, labelpad=10)
        ax.set_zlabel("Solution", fontsize=12, labelpad=5)

        time = np.round(time_step * self.dt,4)
        # Title formatting
        if radius is not None:
             radius = np.round(radius,4)
             if time_step == 0:
                title = f'Initial Solution, with radius={radius}'
             else:
                title = f'Solution at time {time}, with radius={radius}'
        else:
            title = 'Initial Solution' if time_step == 0 else f'Solution at time {time}'

        ax.set_title(title, fontsize=14, pad=15)

        if output_dir is None:
            out_path = f"/wsgjsc/home/wenchel1/dg_nasm/output/baseline/solution_{time_step}.pdf"
        else:
            out_path = os.path.join(output_dir, f"solution_{time_step}.pdf")
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Saved solution to {out_path}")
        
    def plot_analytical_solution(self):
        """
        Plot the solution.
        Plot each element separately.
        Ensure consistent color mapping across all elements.
        """
        # Compute global min and max values of the solution
        global_min = np.min(self.u)
        global_max = np.max(self.u)
        
        # Set up the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Loop over elements
        for elem_idx in range(self.N_elems):
            # Get the coordinates of the element
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof+1, self.dof+1))
            y = np.array(y_coords).reshape((self.dof+1, self.dof+1))
            values = np.array(self.u_analytical[self.element_connectivity[elem_idx]]).reshape((self.dof+1, self.dof+1))
            
            # Plot the element
            surface = ax.plot_surface(x, y, values, cmap='viridis', vmin=global_min, vmax=global_max)
        
        # Add color bar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        fig.suptitle('Analytical Solution')

        # Show the plot
        #plt.show()

    def plot_spatial_error(self):
        """
        Plot the error on the nodes.
        Plot each element separately.
        Ensure consistent color mapping across all elements.
        """
        # Compute global min and max values of the solution
        global_min = np.min(self.u)
        global_max = np.max(self.u)
        
        # Set up the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Loop over elements
        for elem_idx in range(self.N_elems):
            # Get the coordinates of the element
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof+1, self.dof+1))
            y = np.array(y_coords).reshape((self.dof+1, self.dof+1))
            values = np.array((self.e_h[self.element_connectivity[elem_idx]])\
                            .reshape((self.dof+1, self.dof+1)))
            
            # Plot the element
            surface = ax.plot_surface(x, y, values, cmap='viridis', vmin=global_min, vmax=global_max)
        
        # Add color bar
        cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Error Magnitude')
        fig.suptitle('Spatial Error')
        # Show the plot
        plt.tight_layout()
        #plt.show()

    def plot_all_solutions(self, subdomains = 1, overlap = 1, alpha=1, beta=1, symmetry=1, output_dir = None):
        """
        Plot the numerical solution, analytical solution, and spatial error in one figure with 3 subplots.
        """
        # Compute global min and max values of the solution for consistent color mapping
        global_min = np.min(self.u)
        global_max = np.max(self.u)

        fig = plt.figure(figsize=(15, 6))  # Set the overall figure size (smaller height)

        # Numerical solution subplot
        ax1 = fig.add_subplot(131, projection='3d')
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof + 1, self.dof + 1))
            y = np.array(y_coords).reshape((self.dof + 1, self.dof + 1))
            values = np.array(self.u[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            ax1.plot_surface(x, y, values, cmap='viridis', vmin=global_min, vmax=global_max)

        ax1.set_title('Numerical Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')

        # Analytical solution subplot
        ax2 = fig.add_subplot(132, projection='3d')
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof + 1, self.dof + 1))
            y = np.array(y_coords).reshape((self.dof + 1, self.dof + 1))
            values = np.array(self.u_analytical[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            ax2.plot_surface(x, y, values, cmap='viridis', vmin=global_min, vmax=global_max)

        ax2.set_title('Analytical Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')

        # Spatial error subplot
        ax3 = fig.add_subplot(133, projection='3d')
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            x = np.array(x_coords).reshape((self.dof + 1, self.dof + 1))
            y = np.array(y_coords).reshape((self.dof + 1, self.dof + 1))
            values = np.array(self.e_h[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            surface = ax3.plot_surface(x, y, values, cmap='viridis', vmin=global_min, vmax=global_max)

        ax3.set_title('Spatial Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('Error')


        # do a general title for all subplots to print the parameters N, subdomains, alpha, beta, symmetry
        fig.suptitle(f"Numerical Solution, Analytical Solution, and Spatial Error\nN = {self.N_elems}, Overlap = {overlap}, Subdomains = {subdomains}, Alpha = {alpha}, Beta = {beta}, Symmetry = {symmetry}")
        # make sure everything is visible
        plt.tight_layout()
        # Add a colorbar for the third subplot (Spatial Error)
        # cbar = fig.colorbar(surface, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
        # cbar.set_label('Error Magnitude')

        # Adjust layout: increase horizontal space, reduce vertical space, and avoid overlap
        plt.subplots_adjust(wspace=0.5, hspace=0.2)

        if output_dir is None:
            fname = f'output/plots/all_solutions/all_solutions_{self.N_elems}_subdomains{subdomains}_alpha{alpha}_beta{beta}_symmetry{symmetry}.png'
        else:
            fname = f'{output_dir}/all_solutions_{self.N_elems}_subdomains{subdomains}_alpha{alpha}_beta{beta}_symmetry{symmetry}.png'
        plt.savefig(fname, 
                dpi=300, bbox_inches='tight')        
        #plt.show()

        # print where it was stored
        print(f"Spatial error plot saved to: " + fname)

    import numpy as np


    def plot_solution1D(self, y=0.0, time_step=0, output_dir = None):
        """
        Plot the numerical solution along a horizontal line at y = const, and 
        find the first x-value where y = target_y using linear interpolation.

        Parameters:
        y (float): The y-coordinate at which to extract the solution.
        time_step (int): The time step index (for filename).
        tol (float): Tolerance for selecting points near the given y.
        """
        tol = 1e-3
        # Find all points close to y
        mask = np.abs(self.coord[:, 1] - y) < tol
        if not np.any(mask):
            print(f"No points found near y = {y} with tolerance {tol}. Adjust tol if necessary.")
            return

        # Extract x-coordinates and corresponding solution values
        x_values = self.coord[mask, 0]
        u_values = self.u[mask]

        # Keep x-values unsorted (you mentioned sorting isn't needed)
        x_sorted = x_values
        u_sorted = u_values

        # Find the first x where y = 0.5
        x_crossing = self.find_x_for_y(y = 0.0, tol = tol, target_u=0.5)
        x_crossing = x_crossing # flip the sign to get the positive radius

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(x_sorted, u_sorted, 'o', markersize=4)
        plt.xlabel("x")
        plt.ylabel("Solution u")
        
        analytical_radius = self.compute_R(time_step)
        print(f"Analytical determined radius: {analytical_radius}, approximated radius: {x_crossing:.4f}")

        # Include x_crossing in the title if found
        title_text = f"1D Solution Plot at y={y} at time {time_step*self.dt}, \n"
        title_text += f"Analytical radius: {analytical_radius:.4f}, "
        title_text += f"Approximated radius {x_crossing:.4f}"

        plt.title(title_text)
        plt.grid()

        # Store the plot
        if output_dir is None:
            output_dir = "/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/testing"

        output_dir = output_dir + f"/1D_y{y}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fname = f"{output_dir}/y{y}_time{time_step}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"1D solution plot saved to: {fname}")
        return analytical_radius, x_crossing, self.dt*time_step


    def find_x_for_y(self, y = 0.0, tol =  1e-3,  target_u=0.5):
        """
        Find the first x-value corresponding to y = target_y using linear interpolation.

        Parameters:
        y_values (array-like): Array of y-coordinates.
        x_values (array-like): Corresponding array of x-coordinates.
        target_y (float): The y-value to find the corresponding x for.

        Returns:
        float: Interpolated x-value where y = target_y, or None if not found.
        """
        mask = np.abs(self.coord[:, 1] - y) < tol
        if not np.any(mask):
            print(f"No points found near y = {y} with tolerance {tol}. Adjust tol if necessary.")
            return

        # Extract x-coordinates and corresponding solution values
        x_values = self.coord[mask, 0]
        u_values = self.u[mask]

        for i in range(u_values.shape[0] - 1):
            u0, u1 = u_values[i], u_values[i+1]
            x0, x1 = x_values[i], x_values[i+1]

            # Check if y crosses target_y between y0 and y1
            if (u0 <= target_u and u1 >= target_u):
                # Linear interpolation to find corresponding x
                x_interp = x0 + (target_u - u0) * (x1 - x0) / (u1 - u0)
                return -x_interp  # Return first found occurrence

        print("No crossing found for y =", target_u)
        return 0  # Return None if no crossing is found

    def plot_nonlinear_term(self, time_step=0):
        """
        plot the nonlinear term of the initial solution
        """

        pass
        
        
    def plot_solution2D(self, time_step=0, output_dir = None): 
        """
        Plot the numerical solution, analytical solution, and spatial error as 2D colormaps.
        """

        # Compute global min and max values of the solution for consistent color mapping
        global_min = np.min(self.u)
        global_max = np.max(self.u)

        #create figure with just one plot
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            values = np.array(self.u_analytical[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            x = x_coords.reshape((self.dof + 1, self.dof + 1))
            y = y_coords.reshape((self.dof + 1, self.dof + 1))
            mesh = ax1.pcolormesh(x, y, values, cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)
        ax1.set_title('Numerical Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(mesh, ax=ax1, fraction=0.046, pad=0.04)

        # store the plot
        import os
        if output_dir is None:
            output_dir = "/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/"
        
        # Create directory only if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fname = f"{output_dir}/solution_2d_time{time_step}.png"

        # Your existing plot code
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D spatial error plot saved to: {fname}")


    def plot_all_solutions_2d(self, subdomains=1, overlap=1, alpha=1, beta=1, symmetry=1, output_dir = None):
        """
        Plot the numerical solution, analytical solution, and spatial error as 2D colormaps.
        """

        # Compute global min and max values of the solution for consistent color mapping
        global_min = np.min(self.u)
        global_max = np.max(self.u)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))  # Create figure with 3 subplots

        # Numerical solution subplot
        ax1 = axes[0]
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            values = np.array(self.u[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            x = x_coords.reshape((self.dof + 1, self.dof + 1))
            y = y_coords.reshape((self.dof + 1, self.dof + 1))
            mesh = ax1.pcolormesh(x, y, values, cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)

        ax1.set_title('Numerical Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(mesh, ax=ax1, fraction=0.046, pad=0.04)

        # Analytical solution subplot
        ax2 = axes[1]
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            values = np.array(self.u_analytical[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            x = x_coords.reshape((self.dof + 1, self.dof + 1))
            y = y_coords.reshape((self.dof + 1, self.dof + 1))
            mesh = ax2.pcolormesh(x, y, values, cmap='viridis', shading='auto', vmin=global_min, vmax=global_max)

        ax2.set_title('Analytical Solution')
        ax2.set_xlabel('x')
        plt.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.04)

        # Spatial error subplot
        ax3 = axes[2]
        for elem_idx in range(self.N_elems):
            x_coords = self.coord[self.element_connectivity[elem_idx], 0]
            y_coords = self.coord[self.element_connectivity[elem_idx], 1]
            values = np.array(self.e_h[self.element_connectivity[elem_idx]]).reshape((self.dof + 1, self.dof + 1))
            x = x_coords.reshape((self.dof + 1, self.dof + 1))
            y = y_coords.reshape((self.dof + 1, self.dof + 1))
            mesh = ax3.pcolormesh(x, y, values, cmap='viridis', shading='auto')

        ax3.set_title('Spatial Error')
        ax3.set_xlabel('x')
        plt.colorbar(mesh, ax=ax3, fraction=0.046, pad=0.04)

        # Set overall title with parameters
        fig.suptitle(f"Numerical Solution, Analytical Solution, and Spatial Error\nN = {self.N_elems}, Overlap = {overlap}, Subdomains = {subdomains}, Alpha = {alpha}, Beta = {beta}, Symmetry = {symmetry}")

        # Adjust layout to make sure everything is visible
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.2)

        # Save the figure
        if output_dir is None:
            fname = f'output/plots/all_solutions/all_solutions_2d_{self.N_elems}_subdomains{subdomains}_alpha{alpha}_beta{beta}_symmetry{symmetry}.png'
        else:
            fname = f'{output_dir}/all_solutions_2d_{self.N_elems}_subdomains{subdomains}_alpha{alpha}_beta{beta}_symmetry{symmetry}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')        
        #plt.show()

        # Print the file path
        print(f"2D spatial error plot saved to: {fname}")

