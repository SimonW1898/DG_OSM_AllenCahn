# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D



from .mesh import Mesh
from .submesh import SubMesh
from .dg import DG

import datetime
import os

class DomainDecomposition():
    """  
    Class to handle the domain decomposition
    - subdividing
    - solving the subdomains with dg
    - composing the global solution
    """
    def __init__(self, global_number_of_elements = 8, number_of_subdomains = 4,\
                overlap = 1, dof = 1, domain = [0,1], alpha = 1, beta =1,\
                symmetry_parameter = 1, penalty = 18, superpenalisation = False, 
                linear_solver_tol = 1e-14):
        """ 
        Input:
            - global_number_of_elements: int
            - number_of_subdomains: int
            - overlap: int
            - dof: int
        """
        self.number_of_subdomains = number_of_subdomains
        self.overlap = overlap
        self.alpha = alpha
        self.beta = beta
        self.linear_solver_tol = linear_solver_tol
        # set dd iterations
        self.dd_iterations = 0

        # create the global mesh
        self.global_mesh = Mesh(cell_shape=(global_number_of_elements,global_number_of_elements), dof=dof, x_domain=domain, y_domain=domain)

        # create the submeshes
        self.submeshes = self.create_submeshes()

        # set dg parameters
        self.symmetry = symmetry_parameter
        # choose penalty depending on the degree of freedom
        # self.penalty = (self.global_mesh.dof+1)*6
        self.penalty = penalty

        if superpenalisation:
            self.superpenalisation = 3/(self.global_mesh.dof+1)
        else:
            self.superpenalisation = 1
        


        # create the DG objects
        self.dgs = self.create_dgs()

        # solve the subdomains
        # self.solve_subdomains()

        # self.global_mesh.plot_spatial_error()

        # compose the global solution
        # self.compose_global_solution()
        # self.global_mesh.plot_spatial_error()

        # compute the norms
        self.compute_norms()

        self.dd_iterations = np.zeros(self.global_mesh.N_timestep)
        # create meshgrid solution and error
        # self.create_meshgrid_of_solution()

        
    
    def create_submeshes(self):
        """ 
        Create the submeshes
        Output:
            - list of SubMesh objects
        """
        submeshes = []
        for i in range(self.number_of_subdomains**2):
            submesh = SubMesh(self.global_mesh, submesh_idx = i, overlap=self.overlap, number_of_submeshes=self.number_of_subdomains)
            submeshes.append(submesh)
        return submeshes
    
    def create_dgs(self):
        """ 
        Create the DG objects
        Output:
            - list of DG objects
        """
        dgs = []
        for submesh in self.submeshes:
            # print("const", self.beta)
            dg = DG(submesh, symmetry_parameter=self.symmetry, penalty=self.penalty, superpenalisation=self.superpenalisation, linear_solver_tol=self.linear_solver_tol, alpha=self.alpha, beta=self.beta)
            dgs.append(dg)
        return dgs

    def solve_subdomains(self):
        """ 
        Solve the subdomains
        """
        for dg in self.dgs:
            dg.solve_poisson_matrix_free()

    def compose_global_solution(self):
        """  
        compose the global solution from the subdomain solutions
        """
        # init u
        u = np.zeros((self.global_mesh.N_nodes))
        for dg in self.dgs:
            # get the non overlapping elements
            element_indices = dg.get_non_overlapping_elements()

            # get the global indices of the submesh
            global_element_indices = dg.mesh.mapping_subdomain2global_elements.flatten()[element_indices]
            # loop over the elements
            # print("global elem idx",global_element_indices)
            for i in range(global_element_indices.shape[0]):
                # get the global node indices of the element
                node_idx_local = dg.mesh.element_connectivity[element_indices[i]]
                node_idx_global = self.global_mesh.element_connectivity[global_element_indices[i]]

                # set the global solution
                u[node_idx_global] = dg.mesh.u[node_idx_local]
                # print("node idx global", node_idx_global)
                # print("node idx local", node_idx_local)

        # set the global solution
        self.global_mesh.set_u(u)
        # test that global solution was changed at both places
        # print("u change", np.linalg.norm(self.global_mesh.u - self.dgs[0].mesh.parent_mesh.u))

    def compute_norms(self):
        """ 
        Compute the L2 and H0 errors of the global mesh by looping over dg objects and adding up the non overlapping norms
        """
        L2_error = 0
        H0_error = 0

        for dg in self.dgs:
            L2_single, H0_single = dg.compute_error_without_overlap(use_raw = True)
            L2_error += L2_single
            H0_error += H0_single

        normalization = np.linalg.norm(self.global_mesh.u_analytical)**2
        self.L2_error = np.sqrt(L2_error/normalization)
        self.H0_error = np.sqrt(H0_error/normalization)

        return self.L2_error, self.H0_error
    
    def compute_residual(self):
        """ 
        Compute the residual of the global mesh
        """
        residual_vector_list = []
        denominator_vector_list = []
        # loop over the submeshes
        for dg in self.dgs:
            residual_vector_list.append(dg.compute_residual_vector())
            denominator_vector_list.append(dg.compute_denominator_residual())
        # cast to numpy array
        residual_vector = np.concatenate(residual_vector_list)
        denominator_vector = np.concatenate(denominator_vector_list)
        # compute the residual
        residual = np.linalg.norm(residual_vector) / np.linalg.norm(denominator_vector)
        
        return residual

    def compute_residual_with_glob_mat(self):
        """ 
        Compute the residual r = b - Au
        by creating global matrix A and multiplying it with the solution vector u obtained by domain decomposition
        by the otherway the residual is always satisfied to linear solver accuracy after one iteration
        """
        dg_global = DG(self.global_mesh, penalty=self.penalty, symmetry_parameter=self.symmetry, superpenalisation=self.superpenalisation, linear_solver_tol=self.linear_solver_tol, alpha=self.alpha)
        Au = dg_global.compute_Au(self.global_mesh.u)
        rhs = dg_global.compute_rhs()
        residual_vector = rhs - Au

        return np.linalg.norm(residual_vector)


    def additive_schwarz(self, max_iter = 40, max_residual = 1e-8):
        """ 
        Additive Schwarz method
        Input:
            max_iter: (int) maximum number of iterations
        """
        plt.close("all")
        # create output dir 
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        output_dir = f"/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/dd/{time}"
        output_dir_dg = f"/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/dd/dg/{time}"
        # create output dir
        import os
        os.makedirs(output_dir)
        os.makedirs(output_dir_dg)
        # for comparison with pure dg
        import copy
        global_dg = DG(copy.deepcopy(self.global_mesh), symmetry_parameter=self.symmetry, penalty=self.penalty, superpenalisation=self.superpenalisation, linear_solver_tol=self.linear_solver_tol, alpha=self.alpha, beta=self.beta)

        self.plot_both_solutions(y = 0.0, time_step = 0, u_dg = global_dg.mesh.u, output_dir = output_dir)
        N_time = self.global_mesh.N_timestep
        radius_analytical = np.zeros(N_time)
        radius_numerical_dd = np.zeros(N_time)
        radius_numerical_dg = np.zeros(N_time)
        time = np.zeros(N_time)
        for time_step in range(1, N_time + 1):
        # for time_step in range(1, self.global_mesh.N_timestep + 1):

            residual = np.zeros(max_iter+1)
            # set the rhs in each timestep
            for dg in self.dgs:
                dg.mesh.set_rhs(dg.compute_rhs())
                dg.mesh.set_u_old()

            global_dg.mesh.set_rhs(dg.compute_rhs())
            residual[0] = self.compute_residual()
            # compute the rhs in one schwarz iteration
            # self.global_mesh.plot_solution2D(time_step=time_step, output_dir=output_dir)
            for i in range(1,max_iter+1):
                # loop over the submeshes and solve the poisson problem
                plt.figure()
                counter = 0
                for dg in self.dgs:
                    rhs_old = dg.mesh.rhs
                    dg.mesh.set_rhs(dg.update_rhs())
                    
                for dg in self.dgs:
                    dg.allen_cahn_time_step()
                
                # extend the solution to the global mesh
                self.compose_global_solution()

                if  self.compute_residual() < max_residual:
                    if i>1:
                        print("Converged after", i, "iterations", " at time step ", time_step)
                    break
            
            global_dg.mesh.set_rhs(global_dg.compute_rhs())
            global_dg.allen_cahn_time_step()


            print("dd")
            radius_analytical[time_step-1], radius_numerical_dd[time_step-1], time[time_step-1] = self.global_mesh.plot_solution1D(y=0, time_step=time_step, output_dir=output_dir)
            # self.global_mesh.plot_all_solutions_2d(subdomains=self.number_of_subdomains, overlap=self.overlap, alpha=self.alpha, beta=self.beta, symmetry=self.symmetry, output_dir=output_dir)
            print("Pure dg")
            _, radius_numerical_dg[time_step-1], _ = global_dg.mesh.plot_solution1D(y=0.0, time_step=time_step, output_dir=output_dir_dg)
            print("diff of solutions", np.linalg.norm(self.global_mesh.u - global_dg.mesh.u))            
            self.dd_iterations[time_step-1] = i +1

        global_dg.mesh.set_u(self.global_mesh.u - global_dg.mesh.u)
        global_dg.mesh.plot_all_solutions(subdomains=0, overlap=0, alpha=self.alpha, beta=self.beta, symmetry= 0, output_dir=output_dir)

        # plot the radius
        plt.figure()
        plt.plot(time, radius_analytical, label="Analytical")
        plt.plot(time, radius_numerical_dd, label="Numerical DD")
        plt.plot(time, radius_numerical_dg, label="Numerical DG")
        plt.title("Radius of the solution")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.legend()
        plt.savefig(f"{output_dir}/radius.png")
        plt.close()
        print("radius plot saved at", f"{output_dir}/radius.png")
    
    def solve_allen_cahn_dd(self, max_iter=10, max_residual=1e-8, modulo_plot=15):
        """ 
        Solve the Allen-Cahn problem using the Additive Schwarz method and store results in a file.
        Evaluates the solution only at every `modulo_plot` time step.
        """
        def write_radius_data(file_path, metadata=None, values=None, mode="w"):
            """
            Writes radius values and metadata to a file for domain decomposition (DD) and DG cases.
            
            :param file_path: Path to the file
            :param metadata: Dictionary with metadata (only written if mode="w")
            :param values: List of (time_step, radius_analytical, radius_numerical_dd, radius_numerical_dg)
            :param mode: "w" for write (overwrites), "a" for append
            """
            with open(file_path, mode) as f:
                if metadata and mode == "w":
                    f.write("# Metadata\n")
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("# time_step, radius_analytical, radius_numerical_dd, radius_numerical_dg\n")

                if values:
                    for time_step, radius_analytical, radius_numerical_dd, radius_numerical_dg in values:
                        f.write(f"{time_step} {radius_analytical} {radius_numerical_dd} {radius_numerical_dg}\n")

        # Create unique output directory
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        output_dir = f"/wsgjsc/home/wenchel1/dg_nasm/output/plots/allen_cahn/dd/{time_stamp}"
        os.makedirs(output_dir, exist_ok=True)

        output_dir_dg = f"{output_dir}/dg"
        os.makedirs(output_dir_dg, exist_ok=True)

        # For comparison: pure DG method
        import copy
        global_dg = DG(copy.deepcopy(self.global_mesh), symmetry_parameter=self.symmetry, penalty=self.penalty,
                    superpenalisation=self.superpenalisation, linear_solver_tol=self.linear_solver_tol,
                    alpha=self.alpha, beta=self.beta)

        # Initialize radius storage
        N_time = self.global_mesh.N_timestep

        # File to store radius data
        radius_file = f"{output_dir}/radius_data.txt"
        write_radius_data(radius_file, metadata={
            "time_resolution": self.global_mesh.dt, 
            "mesh_size": self.global_mesh.cell_shape,
            "output_directory": output_dir,
            "overlap": self.overlap,
            "stability_parameter": self.penalty,
            "symmetry_parameter": self.symmetry
        }, values=[], mode="w")  # Initialize file

        # Plot initial solution
        print("Initial solution")
        radius_analytical , radius_numerical_dd , time_steps  = self.global_mesh.plot_solution1D(y=0.0, time_step=0, output_dir=output_dir)
        _, radius_numerical_dg , _ = global_dg.mesh.plot_solution1D(y=0.0, time_step=0, output_dir=output_dir_dg)

        # Append initial data to file
        write_radius_data(radius_file, values=[(time_steps , radius_analytical , radius_numerical_dd , radius_numerical_dg )], mode="a")

        self.global_mesh.plot_solution(time_step=0, output_dir=output_dir, radius = radius_numerical_dd)
        # Time stepping loop
        for time_step in range(1, N_time + 1):
            residual = np.zeros(max_iter + 1)

            # Set RHS for each subdomain
            for dg in self.dgs:
                dg.mesh.set_rhs(dg.compute_rhs())
                dg.mesh.set_u_old()

            global_dg.mesh.set_rhs(global_dg.compute_rhs())

            # Schwarz iterations
            for i in range(1, max_iter + 1):
                for dg in self.dgs:
                    dg.mesh.set_rhs(dg.update_rhs())

                for dg in self.dgs:
                    dg.allen_cahn_time_step()

                self.compose_global_solution()  # Extend solution to the global mesh

                if self.compute_residual() < max_residual:
                    if i > 1:
                        print(f"Converged after {i} iterations at time step {time_step}")
                    break
            
            # Solve Allen-Cahn with pure DG method
            global_dg.mesh.set_rhs(global_dg.compute_rhs())
            global_dg.allen_cahn_time_step()

            if time_step % modulo_plot == 0:  # Store results only every `modulo_plot` time steps
                # Store radius values
                
                radius_analytical, radius_numerical_dd, time_steps = self.global_mesh.plot_solution1D(
                    y=0.0, time_step=time_step, output_dir=output_dir)
                _, radius_numerical_dg, _ = global_dg.mesh.plot_solution1D(y=0.0, time_step=time_step, output_dir=output_dir_dg)
                # global_dg.mesh.plot_solution(time_step=time_step, output_dir=output_dir_dg)
                self.global_mesh.plot_solution(time_step=time_step, output_dir=output_dir, radius=radius_numerical_dd)
                # Append results to file
                write_radius_data(radius_file, values=[(time_steps , radius_analytical , radius_numerical_dd , radius_numerical_dg )], mode="a")



        print(f"Radius data saved at {radius_file}")
        # plot diff of dg and dd
        global_dg.mesh.set_u(self.global_mesh.u - global_dg.mesh.u)
        global_dg.mesh.plot_solution(time_step=100000, output_dir=output_dir)

        return radius_file
    
        

    def plot_both_solutions(self,y = 0.0,time_step = 0, u_dg = None, output_dir = None):
        """ 
        Plot both solutions in 1D
        """
        tol = 1e-3
        mask = np.abs(self.global_mesh.coord[:, 1] - y) < tol
        if not np.any(mask):
            print(f"No points found near y = {y} with tolerance {tol}. Adjust tol if necessary.")
            return

        # Extract x-coordinates and corresponding solution values
        x_values = self.global_mesh.coord[mask, 0]
        u_values = self.global_mesh.u[mask]

        u_dg_values = u_dg[mask]

        # Plot the solutions
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, u_values,'-o', label="Domain Decomposition")
        plt.plot(x_values, u_dg_values,'-o', label="DG")
        for i, (x, u) in enumerate(zip(x_values, u_values)):
            plt.text(x, u, str(i), fontsize=12, ha='right', va='bottom')

        # for i, (x, u) in enumerate(zip(x_values, u_dg_values)):
        #     plt.text(x, u, str(i), fontsize=12, ha='left', va='top')

        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Solutions at y = {y}")
        plt.legend()

        if output_dir is None:
            output_dir = "output/plots/"
        plt.savefig(output_dir + f"compare_dd_dg{y}_time_{time_step}.png")
        plt.close()
        print("both solutions plot saved in", output_dir + f"compare_dd_dg{y}_time_{time_step}.png")

    def plot_subdomain_border(self, ax, color, submesh):
        """  
        plot just the boundary faces of the subdomain in ax with color
        """
        # boundary elements 
        # offset 
        offset = 0.01 * (self.global_mesh.x_domain[1] - self.global_mesh.x_domain[0])
        for i in range(4):
            for face_idx in submesh.boundary_cells[i]:
                idx_faces = submesh.get_face_indices(face_idx)
                coord = submesh.coord[idx_faces[i]]
                if i == 0:
                    coord[:,0] -= offset
                elif i == 1:
                    coord[:,0] += offset
                elif i == 2:
                    coord[:,1] -= offset
                elif i == 3:
                    coord[:,1] += offset

                polygon = patches.Polygon(coord, edgecolor=color, facecolor='none', alpha=1, linewidth=5)
                ax.add_patch(polygon)



    def plot_subdomains(self):
        """
        Plot the parent mesh and the different submeshes within
        use patches to plot the subdomains
        """
        # plot the parent mesh
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        self.global_mesh.plot_mesh(ax)
        
        colors = plt.cm.get_cmap('tab10', len(self.submeshes)).colors.tolist()  # Convert to a list

        for submesh in self.submeshes:
            color = colors.pop()
            self.plot_subdomain_border(ax, color, submesh)
        plt.show()
        
    def evaluate_additive_schwarz(self, max_iter = 100, max_residual = 1e-12, plot_bool = False):
        """   
        Function to evaluate the additive Schwarz method in evaluation loop to see how the error decreases 
        and after how many iterations it converges
        """
        H0 = np.zeros(max_iter+1)
        L2 = np.zeros(max_iter+1)
        residuals = np.zeros(max_iter+1)
        self.compute_norms()
        H0[0] = self.H0_error
        L2[0] = self.L2_error
        residuals[0] = self.compute_residual()
        # residuals[0] = self.compute_residual_with_glob_mat()
        e_h_0 = self.e_h_single_meshgrid

        # time loop
        for time_step in range(self.global_mesh.N_timesteps):
            for i in range(max_iter):
                # loop over the submeshes and solve the poisson problem
                for dg in self.dgs:
                    dg.allen_cahn_time_step()
                    # print("dg",dg.beta)
                # extend the solution to the global mesh
                
                self.compose_global_solution()
                self.global_mesh.plot_solution1D()
                self.compute_norms()
                H0[i+1] = self.H0_error
                L2[i+1] = self.L2_error
                residuals[i+1] = self.compute_residual()
                if  residuals[i+1] < max_residual:
                    print("Residual norm:", residuals[i+1])
                    print("Converged after", i+1, "iterations")
                    self.dd_iterations = i+1
                    break

                # stop if diverging
                if L2[i+1] > 1e2:
                    print("Diverging after", i, "iterations")
                    self.dd_iterations = i
                    if i>1:
                        break
                
                
            
        self.dd_iterations = i +1

        if plot_bool:
            print("global spatial error", np.max(np.abs(self.global_mesh.e_h_raw)))
            linewidth = 3
            # Plot solution
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # First subplot: H0 and L2 errors
            ax[0].semilogy(H0[:self.dd_iterations+1], label="H0 error", linewidth=linewidth)
            ax[0].semilogy(L2[:self.dd_iterations+1], label="L2 error", linewidth=linewidth)

            # Add a dotted line for reference DG solution with no domain decomposition
            number_of_elements = self.global_mesh.cell_shape[0]
            dof = self.global_mesh.dof
            domain = self.global_mesh.x_domain
            global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
            dg = DG(global_mesh, symmetry_parameter=self.symmetry, penalty=self.penalty, 
                    superpenalisation=self.superpenalisation, linear_solver_tol=self.linear_solver_tol, 
                    alpha=self.alpha, beta=self.beta)
            dg.solve_poisson_matrix_free()
            L2_dg, H0_dg = dg.compute_error_without_overlap()
            ax[0].axhline(y=H0_dg, color='black', linestyle='--', label="H0 error DG", linewidth=linewidth)
            ax[0].axhline(y=L2_dg, color='black', linestyle='--', label="L2 error DG", linewidth=linewidth)

            ax[0].legend()
            ax[0].set_xlabel("Iterations")
            ax[0].set_ylabel("Error")

            # Second subplot: Residual
            ax[1].semilogy(residuals[:self.dd_iterations+1], label="Residual", color='black', linewidth=linewidth)
            ax[1].axhline(y=max_residual, color='black', linestyle='--', label="Convergence tolerance ", linewidth=linewidth)
            ax[1].legend()
            ax[1].set_xlabel("Iterations")
            ax[1].set_ylabel("Residual")


            # Shared title for both subplots
            fig.suptitle(
                f"Evaluation of Additive Schwarz Method\nOverlap = {self.overlap}, Alpha = {self.alpha}, "
                f"Beta = {self.beta}, Symmetry = {self.symmetry}, Penalty = {self.penalty}\n"
                f"Number of Subdomains = {self.number_of_subdomains}, Global Mesh Size = {number_of_elements}x{number_of_elements} " ,
                fontsize=14
            )

            # Adjust layout for better spacing
            plt.tight_layout(rect=[0, 0, 1, 0.94])  # Reserve space for suptitle

            # Save the plot in output/plots/baseline
            pathname = "output/baseline/"
            filename = f"evaluation_dd_symmetry_{self.symmetry}_penalty_{self.penalty}_overlap_{self.overlap}_alpha_{self.alpha}_beta_{self.beta}.png"
            plt.savefig(pathname + filename, dpi=300)

            plt.show()
            #print where the plot is
            print(f"Plot for dd convergence saved in {pathname + filename}")
            # plot the solution on the domain

            self.global_mesh.plot_all_solutions(subdomains=self.number_of_subdomains, overlap=self.overlap, alpha=self.alpha, beta=self.beta, symmetry=self.symmetry)
            self.global_mesh.plot_all_solutions_2d(subdomains=self.number_of_subdomains, overlap=self.overlap, alpha=self.alpha, beta=self.beta, symmetry=self.symmetry)
        # print the maximum value of the spatial error
        return self.dd_iterations+1, H0[:self.dd_iterations+1], L2[:self.dd_iterations+1], residuals[:self.dd_iterations+1]
        # return i+1, H0[:i+1], L2[:i+1], residuals[:i+1]

    # for frequency analysis
    def create_meshgrid_of_solution(self):
        """   
        create a meshgrid of the solution
        loop over subdomains and elements to assemble a meshgrid of the solution withoud doubling 
        of the overlapping nodes
        as preperation for the frequency analysis via fft2
        """
        # coord = self.global_mesh.
        x = np.linspace(self.global_mesh.x_domain[0], self.global_mesh.x_domain[1], (self.global_mesh.dof)*self.global_mesh.cell_shape[0]+1)
        y = np.linspace(self.global_mesh.y_domain[0], self.global_mesh.y_domain[1], (self.global_mesh.dof)*self.global_mesh.cell_shape[1]+1)

        xv, yv = np.meshgrid(x, y)  # meshgrid for the nodes
        coord_single = np.array([xv.ravel(), yv.ravel()]).T

        # init the solution and error withouht the overlapping nodes
        u_single = np.zeros((coord_single.shape[0]))
        e_h_single = np.zeros((coord_single.shape[0]))
        
        # loop over the elements in the global mesh
        for elem_idx in range(self.global_mesh.N_elems):
            # get the global indices of the non overlapping elements
            single_idx = self.global_mesh.bottom_left_idx_single[elem_idx] + self.global_mesh.single_mapping

            # get the global indices of the overlapping elements
            double_idx = self.global_mesh.element_connectivity[elem_idx] 

            # get the multiplication stencil
            stencil = self.global_mesh.stencils[self.global_mesh.element_position[elem_idx]]
            # multiply the solution with the stencil to interpolate the solution to the non overlapping nodes
            u_single[single_idx] += self.global_mesh.u[double_idx] * stencil
            e_h_single[single_idx] = self.global_mesh.e_h[double_idx] * stencil
        
        # reshape in meshgrid form
        u_single_meshgrid  = u_single.reshape(xv.shape)
        e_h_single_meshgrid = e_h_single.reshape(xv.shape)
        

        self.u_single_meshgrid = u_single_meshgrid
        self.e_h_single_meshgrid = e_h_single_meshgrid
        
    def frequency_analysis(self, e_h_single_meshgrid = None, iteration = 0):
        """   
        Analyse the frequency spectrum of the solution
        use fft2 to compute the frequency spectrum
        """
        # for plots
        x_min, x_max = self.global_mesh.x_domain[0], self.global_mesh.x_domain[1]
        y_min, y_max = self.global_mesh.y_domain[0], self.global_mesh.y_domain[1]

        self.create_meshgrid_of_solution()
        # fft of the solution
        frequency_domain = np.fft.fft2(self.e_h_single_meshgrid)

        # Compute the magnitude spectrum
        magnitude_spectrum_full = np.abs(frequency_domain)

        # Only show the positive frequency components (the negative ones are just the complex conjugates because signal is real    )
        # Take the first half along both axes
        magnitude_spectrum = magnitude_spectrum_full[:magnitude_spectrum_full.shape[0]//2 + 1,\
                                                     :magnitude_spectrum_full.shape[1]//2 + 1]

        # Compute the positive frequencies
        Nx, Ny = e_h_single_meshgrid.shape
        dx = (self.global_mesh.x_domain[1] - self.global_mesh.x_domain[0]) / Nx
        dy = (self.global_mesh.y_domain[1] - self.global_mesh.y_domain[0]) / Ny
        freq_x = np.fft.fftfreq(Nx, d=dx)[:Nx//2]
        freq_y = np.fft.fftfreq(Ny, d=dy)[:Ny//2]
        

        if e_h_single_meshgrid is None: # just plot  for one single error vector
            # Visualize the spatial error and its frequency domain data
            plt.figure(figsize=(12, 6))

            # Original spatial error
            plt.subplot(1, 2, 1)
            plt.imshow(self.u_single_meshgrid, cmap='gray', extent=[x_min, x_max, y_min, y_max], origin='lower')
            plt.title('Spatial Error')
            plt.colorbar()
            # set x and y axis
            plt.xticks(self.global_mesh.x_domain)
            plt.yticks(self.global_mesh.y_domain)

            # Magnitude spectrum (frequency domain)
            plt.subplot(1, 2, 2)
            plt.imshow((magnitude_spectrum), cmap='gray', extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]],origin='lower')
            plt.title('Magnitude Spectrum of Spatial Error')
            plt.colorbar()

            plt.suptitle(f'Frequency Analysis of Spatial Error (Iteration {self.dd_iterations})')
            plt.show()
        else: # plot for multiple error vectors
            
            # fft of e_h_single_meshgrid
            frequency_domain_e_h_0 = np.fft.fft2(e_h_single_meshgrid)

            # magnitude spectrum 
            magnitude_spectrum_e_h_0_full = np.abs(frequency_domain_e_h_0)

            # Only show the positive frequency components
            # Take the first half along both axes
            magnitude_spectrum_e_h_0 = magnitude_spectrum_e_h_0_full[:magnitude_spectrum_e_h_0_full.shape[0]//2 + 1,\
                                                                     :magnitude_spectrum_e_h_0_full.shape[1]//2 + 1]

            # visualize both spatial errors and their frequency domain data
            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plt.imshow(e_h_single_meshgrid, cmap='gray', extent=[x_min, x_max, y_min, y_max],origin='lower')
            plt.title(f'Spatial Error at Iteration {iteration}')
            plt.colorbar()
            # show x and y axis x = self.parent_mesh.x_domain, y = self.parent_mesh.y_domain
            plt.xlim(self.global_mesh.x_domain)
            plt.ylim(self.global_mesh.y_domain)
            # plt.xticks(x)
            # plt.yticks(y)

            plt.subplot(2, 2, 2)
            plt.imshow((magnitude_spectrum_e_h_0), cmap='gray', extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]],origin='lower')
            plt.title(f'Magnitude Spectrum at Iteration {iteration}')
            plt.colorbar()



            plt.subplot(2, 2, 3)
            plt.imshow(self.e_h_single_meshgrid, cmap='gray', extent=[x_min, x_max, y_min, y_max], origin='lower')
            plt.title(f'Spatial Error at Iteration {self.dd_iterations}')
            plt.colorbar()
            # show x and y axis x = self.parent_mesh.x_domain, y = self.parent_mesh.y_domain
          

            plt.subplot(2, 2, 4)
            plt.imshow((magnitude_spectrum), cmap='gray',extent=[freq_x[0], freq_x[-1], freq_y[0], freq_y[-1]], origin='lower')
            plt.title(f'Magnitude Spectrum at Iteration {self.dd_iterations}')
            plt.colorbar()

            plt.suptitle(f'Frequency Analysis of the Error for {self.global_mesh.cell_shape[0]}x{self.global_mesh.cell_shape[1]}  Elements and {self.number_of_subdomains} Subdomains ')
            plt.show()
            
        # store the plot
        # path
        pathname = "plots/"
        filename = f"frequency_analysis_elements_{self.global_mesh.cell_shape[0]}_subdomains_{self.number_of_subdomains}_iterations_{self.dd_iterations}.png"
        
        plt.savefig(pathname + filename)

    def plot_meshgrid_error(self):
        # coord = self.global_mesh.
        x = np.linspace(self.global_mesh.x_domain[0], self.global_mesh.x_domain[1], (self.global_mesh.dof)*self.global_mesh.cell_shape[0]+1)
        y = np.linspace(self.global_mesh.y_domain[0], self.global_mesh.y_domain[1], (self.global_mesh.dof)*self.global_mesh.cell_shape[1]+1)

        xv, yv = np.meshgrid(x, y)  # meshgrid for the nodes
        
        # plot the single solution
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # surface plot
        surface = ax.plot_surface(xv, yv, self.e_h_single_meshgrid, cmap='viridis')
        fig.colorbar(surface)
        plt.show()
