# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D

import pickle

import pandas as pd



from .mesh import Mesh


class SubMesh(Mesh):
    """ 
    Structure to create a submesh from a parent mesh
    
    """
    def __init__(self, parent_mesh, submesh_idx, overlap = 1, number_of_submeshes = 4):
        """ 
        Implemented that parent mesh size is always 2**n
        Input:
            - parent_mesh: Mesh object
            - submesh_idx: int
            - overlap: int
            - number_of_submesh
        """
        self.parent_mesh = parent_mesh
        self.overlap = overlap
        self.number_of_submeshes = number_of_submeshes
        self.submesh_idx = submesh_idx

        # check the position of the submesh
        # corner cases bottom left = 0, bottom right = number_of_submeshes-1, top left = number_of_submeshes**2-number_of_submeshes, top right = number_of_submeshes**2-1
        self.corner_domain = self.set_corner_domain()

        # boundary cases bottom = 0, top = number_of_submeshes-1, left = 0, right = number_of_submeshes-1
        self.boundary_domain = self.set_boundary_domain()

        # cell shape of the submesh
        self.cell_shape = self.set_domain_shape()

        # compute the mask to determine the matrix that contain the indices of the global elements for that subdomain
        self.mask = self.set_mask()

        # needed information for mapping submesh2global
        self.bottom_left_elem_idx_parent_mesh = self.get_bottom_left_elem_idx_parent_mesh()

        # mapping local to global elements in subdomain
        self.mapping_subdomain2global_elements = self.get_mapping_subdomain2global_elements()

        # set the domain boundaries
        self.x_domain, self.y_domain = self.get_domain()

        # initialize the mesh
        super().__init__(cell_shape = self.cell_shape, dof = self.parent_mesh.dof, x_domain = self.x_domain, y_domain = self.y_domain,)

        # correct the changed attributes by the initialization of the parent mesh
        self.parent_mesh = parent_mesh
        
        # set the is_exterior_boundary
        self.is_exterior_boundary = self.get_is_exterior_boundary()


    def set_corner_domain(self):
        """ 
        Check the position of the submesh with the help of the submesh index and the number of submeshes
        """
        if self.submesh_idx == 0:
            return 0
        elif self.submesh_idx == self.number_of_submeshes-1:
            return 1
        elif self.submesh_idx == self.number_of_submeshes**2-self.number_of_submeshes:
            return 2
        elif self.submesh_idx == self.number_of_submeshes**2-1:
            return 3
        else:
            return -1
    
    def set_boundary_domain(self):
        """  
        Set the index for the position of the subdomains position within the parent mesh
        Output:
            - position: int 
            0: bottom, 1: top, 2: left, 3: right, -1: interior domain -2: corner domain
            0: left, 1: right, 2: bottom, 3: top, -1: interior domain -2: corner domain
        """
        if self.submesh_idx < self.number_of_submeshes:
            if self.corner_domain == 0 or self.corner_domain == 1:
                return -2
            else:
                return 2
        elif self.submesh_idx >= self.number_of_submeshes**2-self.number_of_submeshes:
            if self.corner_domain == 2 or self.corner_domain == 3:
                return -2
            else:
                return 3
        elif self.submesh_idx % self.number_of_submeshes == 0:
            if self.corner_domain == 0 or self.corner_domain == 2:
                return -2
            else:
                return 0
        elif self.submesh_idx % self.number_of_submeshes == self.number_of_submeshes-1:
            if self.corner_domain == 1 or self.corner_domain == 3:
                return -2
            else:
                return 1
        else:
            return -1

    def set_domain_shape(self):
        """
        Set the shape of the submesh 
        Output:
            - cell_shape: tuple (int,int)
        """
        # corner cases
        if not(self.corner_domain == -1):
            return (self.parent_mesh.cell_shape[0]//self.number_of_submeshes + self.overlap, \
                    self.parent_mesh.cell_shape[1]//self.number_of_submeshes + self.overlap)
        
        # boundary cases
        if not(self.boundary_domain == -1):
            # vertical side
            if self.boundary_domain == 0 or self.boundary_domain == 1:
                return (self.parent_mesh.cell_shape[0]//self.number_of_submeshes + self.overlap, \
                    self.parent_mesh.cell_shape[1]//self.number_of_submeshes + 2*self.overlap)
            # horizontal side
            else:
                return (self.parent_mesh.cell_shape[0]//self.number_of_submeshes + 2*self.overlap, \
                        self.parent_mesh.cell_shape[1]//self.number_of_submeshes + self.overlap)
                
            
        # interior case
        if self.corner_domain == -1 and self.boundary_domain == -1:
            return (self.parent_mesh.cell_shape[0]//self.number_of_submeshes + 2*self.overlap, \
                    self.parent_mesh.cell_shape[1]//self.number_of_submeshes + 2*self.overlap)

    def set_mask(self):
        """  
        Set the mask for the submesh to get the indices of the global elements for that subdomain
        dependent on corner, boundary or interior case
        Output:
            - mask: np.array (int)
        """
        mask = np.zeros((self.cell_shape[1],self.cell_shape[0]),dtype=int)
        # CORNER CASES
        # bottom left corner
        if self.corner_domain == 0:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j
        # bottom right corner
        elif self.corner_domain == 1:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.overlap
        # top left corner
        elif self.corner_domain == 2:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap
        # top right corner
        elif self.corner_domain == 3:
            for i in range(self.cell_shape[0]):
                for j in range(self.cell_shape[1]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap - self.overlap
        # BOUNDARY CASES
        # left boundary
        elif self.boundary_domain == 0:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap
        # right boundary
        elif self.boundary_domain == 1:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap - self.overlap
        # bottom boundary
        elif self.boundary_domain == 2:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.overlap
        # top boundary
        elif self.boundary_domain == 3:
            for i in range(self.cell_shape[1]):
                for j in range(self.cell_shape[0]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap - self.overlap
        # INTERIOR CASE
        else:
            for i in range(self.cell_shape[0]):
                for j in range(self.cell_shape[1]):
                    mask[i,j] = i*self.parent_mesh.cell_shape[0] + j - self.parent_mesh.cell_shape[0]*self.overlap - self.overlap
        # print(mask)
        
        return mask
    
    def get_bottom_left_elem_idx_parent_mesh(self):
        """ 
        Get the bottom left element index of the parent mesh that is the bottom left element of the submesh
        """

        mesh_elements = np.arange(self.parent_mesh.N_elems).reshape(self.parent_mesh.cell_shape[1],self.parent_mesh.cell_shape[0])
        
        x_idx = self.parent_mesh.cell_shape[0]//self.number_of_submeshes
        y_idx = self.parent_mesh.cell_shape[1]//self.number_of_submeshes

        x_factor = self.submesh_idx % self.number_of_submeshes
        y_factor = self.submesh_idx // self.number_of_submeshes

        bottom_left_elem_idx = mesh_elements[y_idx*y_factor,x_idx*x_factor]

        return bottom_left_elem_idx

    def get_mapping_subdomain2global_elements(self):
        """  
        add mask with bottom left index to get the global indices of the elements that describe the domain
        """
        mapping = self.mask + self.bottom_left_elem_idx_parent_mesh
        # print(mapping)
        return mapping

    def get_domain(self):
        """ 
        get the domain of the submesh needed for super constructor
        """   
        bottom_left = self.parent_mesh.coord[self.parent_mesh.element_connectivity[self.mapping_subdomain2global_elements[0][0]]][0]
        top_right = self.parent_mesh.coord[self.parent_mesh.element_connectivity[self.mapping_subdomain2global_elements[-1][-1]]][-1]
        x_domain = [bottom_left[0],top_right[0]]
        y_domain = [bottom_left[1],top_right[1]]

        return x_domain, y_domain

    def get_is_exterior_boundary(self):
        """  
        check if the subdomain has exterior boundaries
        """
        if not(self.boundary_domain == -1):
            # corner cases
            if self.boundary_domain == -2:
                if self.corner_domain == 0:
                    return np.array([True,False,True,False])
                elif self.corner_domain == 1:
                    return np.array([False,True,True,False])
                elif self.corner_domain == 2:
                    return np.array([True,False,False,True])
                elif self.corner_domain == 3:
                    return np.array([False,True,False,True])
            # boundary cases
            # left
            elif self.boundary_domain == 0:
                return np.array([True,False,False,False])
            # right
            elif self.boundary_domain == 1:
                return np.array([False,True,False,False])
            # bottom
            elif self.boundary_domain == 2:
                return np.array([False,False,True,False])
            # top
            elif self.boundary_domain == 3:
                return np.array([False,False,False,True])
        # INTERIOR CASE
        else:
            return np.array([False,False,False,False])
            
