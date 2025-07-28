import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


from .domain_decomposition import DomainDecomposition
from .dg import DG
from .mesh import Mesh



def evaluate_convergence_increasing_subdomains(overlap = 1, alpha = 1, number_of_elements_power = 5,  max_iter=50, dof=2, domain=[0,1],linear_solver_tol = 1e-14, dd_residual_tol = 1e-16):
    """    
    double the number of subdomains should take double the iterations
    """
    number_of_elements = 2**number_of_elements_power
    # array with powers of 2 with length number_of_elements_power - 2
    number_of_runs = number_of_elements_power - 2
    if number_of_runs < 1:
        number_of_runs = number_of_elements_power
    number_of_subdomains = [2**i for i in range(1, number_of_runs)]


    max_residual = 4*((domain[1] - domain[0])/(number_of_elements)) ** (dof+1)
    max_residual = dd_residual_tol
    print("Max residual", max_residual)


    H0_all = []
    L2_all = []
    iterations = []
    residuals_all = []
    max_residuals = []
    # loop over the number of subdomains
    for n_subdomains in number_of_subdomains:
        print("Number of subdomains:", n_subdomains)
        dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = n_subdomains, overlap = overlap, dof = dof, domain = domain,  alpha = alpha)
        i, H0, L2, residuals = dd.evaluate_additive_schwarz(max_iter = max_iter, max_residual = max_residual)
        # dd.global_mesh.plot_spatial_error()
        # append the results
        H0_all.append(H0)
        L2_all.append(L2)
        iterations.append(i)
        residuals_all.append(residuals)
        max_residuals.append(max_residual)
    # solve just with dg and without domain decomposition
    global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
    dg_global = DG(global_mesh, symmetry_parameter=dd.symmetry, penalty=dd.penalty, superpenalisation=dd.superpenalisation, linear_solver_tol=1e-12, alpha=alpha, beta=0)
    dg_global.solve_poisson_matrix_free()
    dg_L2, dg_H0 = dg_global.compute_error_with_overlap()
    dg_residual = dg_global.compute_residual()

    # store the results with savez also store number of elements and subdomains nof in the filename also
    fname_prefix = f"output/data/"
    filename = f"convergence_results_{number_of_elements}_{number_of_subdomains}_overlap_{overlap}.pkl"

    path_name = fname_prefix + filename
    with open(path_name, 'wb') as f:
        pickle.dump((number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlap, alpha, dg_L2, dg_H0, dg_residual), f)

    print(f"Results saved in {path_name}")

    # plot the results by calling the plot function
    plot_convergence_results(filename)

    return filename


# evaluate convergence increasing subdomains for different overlaps
def evaluate_convergence_overlap(alpha = 1, max_overlap = 3, number_of_elements_power = 5,max_iter = 50, dof = 2, domain = [0,1], linear_solver_tol = 1e-14, dd_residual_tol = 1e-16):
    """ 
    Evaluate the convergence for increasing number of subdomains with different overlaps to see the 
    effect of the overlap in the convergence
    """
    # array of overlaps
    overlaps = np.arange(max_overlap+1)
    number_of_elements = 2**number_of_elements_power
    number_of_subdomains = 2**(number_of_elements_power-2)
    

    max_residual = 3*((domain[1] - domain[0])/(number_of_elements)) ** (dof+1)
    max_residual = dd_residual_tol
    print("Max residual", max_residual)


    H0_all = []
    L2_all = []
    iterations = []
    residuals_all = []
    max_residuals = []

    # loop over overlaps 
    for overlap in overlaps:
        print("Overlap:", overlap)
        dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = number_of_subdomains, overlap = overlap, dof = dof, domain = domain,  alpha = alpha, symmetry_parameter = -1, linear_solver_tol = linear_solver_tol)
        i, H0, L2, residuals = dd.evaluate_additive_schwarz(max_iter = max_iter, max_residual = max_residual)
        
        H0_all.append(H0)
        L2_all.append(L2)
        iterations.append(i)
        residuals_all.append(residuals)
        max_residuals.append(max_residual)
        
    # solve just with dg and without domain decomposition
    global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
    # mesh = dd.dgs[0].mesh
    dg_global = DG(global_mesh, symmetry_parameter=dd.symmetry, penalty=dd.penalty, superpenalisation=dd.superpenalisation, linear_solver_tol=dd.linear_solver_tol, alpha=alpha, beta=0)
    dg_global.solve_poisson_matrix_free()
    dg_L2, dg_H0 = dg_global.compute_error_without_overlap()
    dg_residual = dg_global.compute_residual()
    dg_residual = max_residual

    # store the results
    fname_prefix = f"output/data/"
    filename = f"convergence_overlaps_{max_overlap}_number_of_elements_{number_of_elements}_subdomains_{number_of_subdomains}.pkl"
    
    pathname = fname_prefix + filename
    with open(pathname, 'wb') as f:
        pickle.dump((number_of_elements, number_of_subdomains,  iterations, H0_all, L2_all, residuals_all, overlaps, alpha, dg_L2, dg_H0, dg_residual), f)
    print(f"Results saved in {filename}")

    # plot the results by calling the plot function
    plot_convergence_overlaps(filename)

    return filename

def evaluate_convergence_alpha(number_of_alpha = 5, overlap = 0, number_of_elements_power = 5, max_iter = 50, dof = 2, domain = [0,1]):
    """ 
    Evaluate the convergence for different values of alpha
    """
    alphas = np.linspace(0,1,number_of_alpha)
    # alphas = np.linspace(0,1,number_of_alpha)

    number_of_elements = 2**number_of_elements_power
    number_of_subdomains = 2**(number_of_elements_power-2)
    

    max_residual = 3*((domain[1] - domain[0])/(number_of_elements)) ** (dof+1)
    max_residual = 1e-8
    print("Max residual", max_residual)

    
    H0_all = []
    L2_all = []
    iterations = []
    residuals_all = []
    max_residuals = []

    # loop over overlaps 
    for alpha in alphas:
        print("Alpha:", alpha)
        dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = number_of_subdomains, overlap = overlap, dof = dof, domain = domain,  alpha = alpha)
        i, H0, L2, residuals = dd.evaluate_additive_schwarz(max_iter = max_iter, max_residual = max_residual)
        
        H0_all.append(H0)
        L2_all.append(L2)
        iterations.append(i)
        residuals_all.append(residuals)
        max_residuals.append(max_residual)

    # solve just with dg and without domain decomposition
    global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
    dg_global = DG(global_mesh, symmetry_parameter=dd.symmetry, penalty=dd.penalty, superpenalisation=dd.superpenalisation, linear_solver_tol=dd.linear_solver_tol, alpha=alpha, beta=0)
    dg_global.solve_poisson_matrix_free()
    dg_L2, dg_H0 = dg_global.compute_error_with_overlap()
    dg_residual = dg_global.compute_residual()

    fname_prefix = f"output/data/"
    filename = f"convergence_alpha_{number_of_alpha}_number_of_elements_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
    
    pathname = fname_prefix + filename
    with open(pathname, 'wb') as f:
        pickle.dump((number_of_elements, number_of_subdomains,  iterations, H0_all, L2_all, residuals_all, overlap, alphas, dg_L2, dg_H0, dg_residual), f)
    print(f"Results saved in {pathname}")

    # plot the results by calling the plot function
    plot_convergence_alphas(filename)

    return filename

# load the results
def load_results(filename):
    """ 
    Load the results of the convergence test
    """
    with open(filename, 'rb') as f:
        number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlaps, alphas, dg_L2, dg_H0, dg_residual= pickle.load(f)
    return number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlaps, alphas, dg_L2, dg_H0, dg_residual


def plot_convergence_results(filename):
    """ 
    Plot and save the convergence results
    """
    fname_prefix_data = f"output/data/"
    # Load results from file
    number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlap, alphas, dg_L2, dg_H0, dg_residual = load_results(fname_prefix_data + filename)

    # Create the H0 and L2 error plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(len(number_of_subdomains)):
        ax[0].semilogy(H0_all[i], label="H0 error, " + str(number_of_subdomains[i]) + " subdomains")
        ax[1].semilogy(L2_all[i], label="L2 error, " + str(number_of_subdomains[i]) + " subdomains")
        ax[2].semilogy(residuals_all[i], label="Residual, " + str(number_of_subdomains[i]) + " subdomains")

    ax[0].axhline(dg_H0, color='black', linestyle='--', label="Error for global mesh DG")
    ax[1].axhline(dg_L2, color='black', linestyle='--', label="Error for global mesh DG")
    ax[2].axhline(dg_residual, color='black', linestyle='--', label="Residual for global mesh DG")
    
    ax[0].legend()
    ax[0].set_title("H0 error")
    ax[1].legend()
    ax[1].set_title("L2 error")
    ax[2].legend()
    ax[2].set_title("Residual")
    
    norm_plot_fname = f'output/plots/increasing_subdomain_H0_L2_' + filename.split("/")[-1].replace(".pkl", ".pdf")
    # Save the H0 and L2 error plot
    plt.savefig(norm_plot_fname,
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close()
    print("Norm plot saved in", norm_plot_fname)

    # Create the residual plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(len(number_of_subdomains)):
        ax.semilogy(residuals_all[i], label="Residual, " + str(number_of_subdomains[i]) + " subdomains")
    
    ax.legend()
    ax.set_title("Residual")
    
    residual_plot_fname = f'output/plots/increasing_subdomain_residual_' + filename.split("/")[-1].replace(".pkl", ".pdf")
    # Save the residual plot
    plt.savefig(residual_plot_fname, 
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    print("Residual plot saved in", residual_plot_fname)
    plt.close()

def plot_convergence_overlaps(filename):
    """ 
    Plot and save the convergence results
    """
    # data prefix
    fname_prefix_data = f"output/data/"
    # Load results from file
    number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlaps, alphas, dg_L2, dg_H0, dg_residual = load_results(fname_prefix_data + filename)
    linewidth = 5
    # Create the H0 and L2 error plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(len(overlaps)):
        ax[0].semilogy(H0_all[i], label="H0 error, " + str(overlaps[i]) + " overlap", linewidth=linewidth)
        ax[1].semilogy(L2_all[i], label="L2 error, " + str(overlaps[i]) + " overlap", linewidth=linewidth)
        ax[2].semilogy(residuals_all[i], label="Residual, " + str(overlaps[i]) + " overlap", linewidth=linewidth)

    ax[0].axhline(dg_H0, color='black', linestyle='--', label="Error for global mesh DG")
    ax[1].axhline(dg_L2, color='black', linestyle='--', label="Error for global mesh DG")
    ax[2].axhline(dg_residual, color='black', linestyle='--', label="Minimal residual for domain decomposition")
    
    ax[0].legend()
    ax[0].set_title("H0 error")
    ax[1].legend()
    ax[1].set_title("L2 error")
    ax[2].legend()
    ax[2].set_title("Residual")
    
    norm_plot_fname = f'output/plots/overlaps_H0_L2_' + filename.split("/")[-1].replace(".pkl", ".pdf")
    # Save the H0 and L2 error plot
    plt.savefig(norm_plot_fname,
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close()
    print("Plot saved in", norm_plot_fname)

def plot_convergence_alphas(filename):
    # data prefix
    fname_prefix_data = f"output/data/"
    # Load results from file
    number_of_elements, number_of_subdomains, iterations, H0_all, L2_all, residuals_all, overlaps, alphas, dg_L2, dg_H0, dg_residual = load_results(fname_prefix_data + filename)
    
    # Create the H0 and L2 error plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(len(alphas)):
        ax[0].semilogy(H0_all[i], label=r"H0 error, $\alpha =$ " + str(alphas[i]))
        ax[1].semilogy(L2_all[i], label=r"L2 error, $\alpha =$ " + str(alphas[i]))
        ax[2].semilogy(residuals_all[i], label=r"Residual, $\alpha =$ " + str(alphas[i]))

    ax[0].axhline(dg_H0, color='black', linestyle='--', label="Error for global mesh DG")
    ax[1].axhline(dg_L2, color='black', linestyle='--', label="Error for global mesh DG")
    ax[2].axhline(dg_residual, color='black', linestyle='--', label="Residual for global mesh DG")
    
    ax[0].legend()
    ax[0].set_title("H0 error")
    ax[1].legend()
    ax[1].set_title("L2 error")
    ax[2].legend()
    ax[2].set_title("Residual")
    
    norm_plot_fname = f'output/plots/alphas_H0_L2_' + filename.split("/")[-1].replace(".pkl", ".pdf")
    # Save the H0 and L2 error plot
    plt.savefig(norm_plot_fname,
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close()
    print("Norm plot saved in", norm_plot_fname)

    # Create the residual plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(len(alphas)):
        ax.semilogy(residuals_all[i], label=r"Residual, $\alpha =$ " + str(alphas[i]))
    
    ax.legend()
    ax.set_title("Residual")
    
    residual_plot_fname = f'output/plots/alphas_residual_' + filename.split("/")[-1].replace(".pkl", ".pdf")
    # Save the residual plot
    plt.savefig(residual_plot_fname, 
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    print("Residual plot saved in", residual_plot_fname)
    plt.close()





# make function to evaluate everything decided by input

def eval_convergence_dd(evaltype = "overlap", n_eval = 3,\
                        overlap = 1, alpha = 0.5, beta = 1, \
                        number_of_elements_power = 4, number_of_subdomains = 4,\
                        dof = 2, domain = [0,1], 
                        symmetry = 1, penalty = 1, superpenalisation = False,
                        max_iter = 50, linear_solver_tol = 1e-12, dd_tol = 1e-10):
    """ 
    Function to evaluate the convergence of the domain decomposition method
    Input:
    - evaltype: string, type of evaluation, either "overlap", "subdomains" or "alpha"
    - n_eval: int, number of evaluations so for overlap 3 means 0,1,2, for subdomains 3 means 2,4,8 etc.
    - rest is clear
    """
    if evaltype == "overlap":
        iterable = np.arange(n_eval+1)

    elif evaltype == "subdomains":
        iterable = 2**np.arange(1, n_eval + 1+ 1)

    elif evaltype == "alpha":
        # Define the ranges and refinements
        start, end = 0.1, 1.5
        iterable = np.linspace(0.01, 1, n_eval)
        add = 3
        iterable = np.logspace(-(n_eval+add), -(1+add), n_eval)
        print(iterable)
        # round to 1 decimal
        # iterable = np.round(iterable, 3)


    elif evaltype == "beta":
        iterable = np.linspace(0.01,15,n_eval)
        iterable = np.round(iterable, 3)
        iterable[1] = 1
        print(iterable)

    
    elif evaltype == "penalty":
        iterable = np.linspace(1, 36, n_eval)

    elif evaltype == "symmetry":
        iterable = np.array([-1,0, 1])
        
    elif evaltype == "superpenalisation":
        iterable = np.array([True, False])

    number_of_elements = 2**number_of_elements_power    

    print("Max residual", dd_tol)

    
    H0_all = []
    L2_all = []
    iterations = []
    residuals_all = []
    max_residuals = []

    # loop over iterable
    for iter in iterable:
        if evaltype == "overlap":
            overlap = iter
        elif evaltype == "subdomains":
            number_of_subdomains = iter
        elif evaltype == "alpha":
            alpha = iter
        elif evaltype == "beta":
            beta = iter
        elif evaltype == "penalty":
            penalty = iter
        elif evaltype == "symmetry":
            symmetry = iter
        elif evaltype == "superpenalisation":
            superpenalisation = iter
            
        print(f"{evaltype}: {iter}")
        dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = number_of_subdomains,\
                                overlap = overlap, dof = dof, domain = domain,  \
                                alpha = alpha, beta=beta, \
                                symmetry_parameter = symmetry, penalty = penalty, superpenalisation = superpenalisation, 
                                linear_solver_tol = linear_solver_tol)
        
        i, H0, L2, residuals = dd.evaluate_additive_schwarz(max_iter = max_iter, max_residual = dd_tol, plot_bool = True)
        
        H0_all.append(H0)
        L2_all.append(L2)
        iterations.append(i)
        residuals_all.append(residuals)
        max_residuals.append(dd_tol)

        # solve just with dg and without domain decomposition
        global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
        dg_global = DG(global_mesh, symmetry_parameter=dd.symmetry, penalty=dd.penalty, superpenalisation=dd.superpenalisation, linear_solver_tol=dd.linear_solver_tol, alpha=alpha, beta=0)
        dg_global.solve_poisson_matrix_free()
        dg_L2, dg_H0 = dg_global.compute_error_without_overlap()


        # filename depending on evaltype
        if evaltype == "overlap":
            filename = f"overlaps_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            overlap = iterable
        elif evaltype == "subdomains":
            filename = f"subdomains_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            number_of_subdomains = iterable
        elif evaltype == "alpha":
            filename = f"alpha_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            alpha = iterable
        elif evaltype == "beta":
            filename = f"beta_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            beta = iterable
        elif evaltype == "penalty":
            filename = f"penalty_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            penalty = iterable
        elif evaltype == "symmetry":
            filename = f"symmetry_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            symmetry = iterable
        elif evaltype == "superpenalisation":
            filename = f"superpenalisation_elem_{number_of_elements}_subdomains_{number_of_subdomains}_overlap_{overlap}.pkl"
            superpenalisation = iterable

    fname_prefix = f"output/data/"
    pathname = fname_prefix + filename

    with open(pathname, 'wb') as f:
        pickle.dump((number_of_elements, number_of_subdomains,  iterations,\
                    H0_all, L2_all, residuals_all,\
                    overlap, alpha, beta,\
                    symmetry, penalty, superpenalisation,\
                    dg_L2, dg_H0, dd_tol)\
                    , f)
    
    print(f"Results saved in {filename}")

    # plot the results by calling the plot function
    plot_convergence_dd(filename, evaltype, dof = dof)
    
    return filename

def load_results_dd(filename):
    """ 
    Load the results of the convergence test
    """
    with open(filename, 'rb') as f:
        number_of_elements, number_of_subdomains,  iterations,\
        H0_all, L2_all, residuals_all,\
        overlap, alpha, beta,\
        symmetry, penalty, superpenalisation,\
        dg_L2, dg_H0, dd_tol = pickle.load(f)

    return  number_of_elements, number_of_subdomains,  iterations,\
            H0_all, L2_all, residuals_all,\
            overlap, alpha, beta,\
            symmetry, penalty, superpenalisation,\
            dg_L2, dg_H0, dd_tol



def plot_convergence_dd(filename, evaltype, dof = 2):
    """
    plot the convergence results depending on the evaltype
    Input:
    - filename: str, filename of the results
    - evaltype: str, type of evaluation,  "overlap", "subdomains" or "alpha"
    """
    linewidth = 5
    title_fontsize = 15

    fname_prefix_data = f"output/data/"
    # Load results from file
    number_of_elements, number_of_subdomains, iterations, \
        H0_all, L2_all, residuals_all, \
        overlap, alpha, beta, \
        symmetry, penalty, superpenalisation, \
        dg_L2, dg_H0, dd_tol = load_results_dd(fname_prefix_data + filename)
    
    # Create the H0 and L2 error plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    if evaltype == "overlap":
        iterable = overlap 
        label_str = "Overlap = "
    elif evaltype == "subdomains":
        iterable = number_of_subdomains 
        label_str = "Number of subdomains = "
    elif evaltype == "alpha":
        iterable = alpha 
        label_str = r"$\alpha =$"
    elif evaltype == "beta":
        iterable = beta 
        label_str = r"$\beta =$"
    elif evaltype == "penalty":
        iterable = penalty 
        label_str = "Penalty = "
    elif evaltype == "symmetry":
        iterable = symmetry 
        label_str = "Symmetry = "
    elif evaltype == "superpenalisation":
        iterable = superpenalisation
        label_str = "Superpenalisation = "

    # Generate a colormap with enough unique colors
    colormap = plt.cm.get_cmap('tab20', len(iterable))  

    for i, iter in enumerate(iterable):
        if evaltype == "symmetry":
            if iter == -1:
                label_str = r"SIPG, $\epsilon =$"
            elif iter == 0:
                label_str = r"NIPG, $\epsilon =$"
            elif iter == 1:
                label_str = r"IIPG, $\epsilon =$"
        color = colormap(i)  # Get a unique color for each line
        ax[0].semilogy(H0_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
        ax[1].semilogy(L2_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
        ax[2].semilogy(residuals_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
    
    ax[0].axhline(dg_H0, color='black', linestyle='--', label="Error for global mesh DG")
    ax[1].axhline(dg_L2, color='black', linestyle='--', label="Error for global mesh DG")
    ax[2].axhline(dd_tol, color='black', linestyle='--', label="Minimal residual for domain decomposition")
    
    ax[0].legend()
    ax[0].set_title(r"$H^0$ broken gradient seminorm of the error", fontsize=title_fontsize)
    ax[1].set_title(r"$L^2$-norm of the error", fontsize=title_fontsize)
    ax[2].set_title(r"$L^2$-norm of the Residual", fontsize=title_fontsize)
    
    # Adjust spacing between subplots and the suptitle
    fig.suptitle(
    f"overlap = {overlap}, alpha = {alpha}, beta = {beta}, symmetry = {symmetry}\n"
    f"penalty = {penalty}, number of elements = {number_of_elements}, polynomial degree = {dof} \n",
    fontsize=title_fontsize,
    y=0.95  # Adjust the vertical position of the suptitle
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle

    # Make dir at output/plots named elements_number_of_elements
    if symmetry == -1:
        dg_class = "SIPG"
    elif symmetry == 0:
        dg_class = "NIPG"
    elif symmetry == 1:
        dg_class = "IIPG"
        
    plot_dir = f"output/plots/elem_{number_of_elements}/"+dg_class+"/"+f"overlap_{overlap}/"
    os.makedirs(plot_dir, exist_ok=True)

    norm_plot_fname = plot_dir + filename.split("/")[-1].replace(".pkl", ".pdf")

    # Save the H0 and L2 error plot
    plt.savefig(norm_plot_fname,
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close()
    print("Norm plot saved in", norm_plot_fname)


def plot_grid(filename):
    """
    do the same plot but for the gridsearch
    """
    linewidth = 5
    title_fontsize = 15

    fname_prefix_data = f"output/data/"
    # Load results from file
    number_of_elements, number_of_subdomains, iterations, \
        H0_all, L2_all, residuals_all, \
        overlap, alpha, beta, \
        symmetry, penalty, superpenalisation, \
        dg_L2, dg_H0, dd_tol = load_results_dd(fname_prefix_data + filename)
    
    # Create the H0 and L2 error plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    iterable = alpha 
    label_str = r"$\alpha =$"
    

    # Generate a colormap with enough unique colors
    colormap = plt.cm.get_cmap('tab20', len(iterable))  

    for i, iter in enumerate(iterable):
        color = colormap(i)  # Get a unique color for each line
        ax[0].semilogy(H0_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
        ax[1].semilogy(L2_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
        ax[2].semilogy(residuals_all[i], label=label_str + str(iter), linewidth=linewidth, color=color)
    
    ax[0].axhline(dg_H0, color='black', linestyle='--', label="Error for global mesh DG")
    ax[1].axhline(dg_L2, color='black', linestyle='--', label="Error for global mesh DG")
    ax[2].axhline(dd_tol, color='black', linestyle='--', label="Minimal residual for domain decomposition")
    
    ax[0].legend()
    ax[0].set_title(r"$H^0$ broken gradient seminorm of the error", fontsize=title_fontsize)
    ax[1].set_title(r"$L^2$-norm of the error", fontsize=title_fontsize)
    ax[2].set_title(r"$L^2$-norm of the Residual", fontsize=title_fontsize)
    
    plt.tight_layout()

    # Make dir at output/plots named elements_number_of_elements
    if symmetry == -1:
        dg_class = "SIPG"
    elif symmetry == 0:
        dg_class = "NIPG"
    elif symmetry == 1:
        dg_class = "IIPG"
        
    fig.suptitle(dg_class + " with overlap = " + str(overlap) + ", number of elements = " + str(number_of_elements)+"\n", fontsize=title_fontsize)
    plot_dir = f"output/plots/grid/elem_{number_of_elements}/"+dg_class+"/"+f"overlap_{overlap}/"
    os.makedirs(plot_dir, exist_ok=True)

    norm_plot_fname = plot_dir + filename.split("/")[-1].replace(".pkl", ".pdf")

    # Save the H0 and L2 error plot
    plt.savefig(norm_plot_fname,
                dpi=300, bbox_inches='tight')
    
    # Close the plot to free memory
    plt.close()
    print("Norm plot saved in", norm_plot_fname)


def eval_all(n_eval = 3,\
                overlap = 1, alpha = 0.5, beta = 1, \
                number_of_elements_power = 4, number_of_subdomains = 4,\
                dof = 2, domain = [0,1], 
                symmetry = 1, penalty = 1, superpenalisation = False,
                max_iter = 50, linear_solver_tol = 1e-12, dd_tol = 1e-10):
    # evaltype_list = ["overlap", "subdomains", "alpha", "beta", "penalty", "symmetry", "superpenalisation"]
    evaltype_list = ["alpha","beta", "penalty"]
    # evaltype_list = ["beta"]
    evaltype_list = ["alpha"]
    # symmetry_params = [-1,0,1]
    symmetry_params = [-1]
    # loop over all evaluation types
    for symmetry in symmetry_params:
        for evaltype in evaltype_list:
            eval_convergence_dd(evaltype = evaltype, n_eval = n_eval,\
                            overlap = overlap, alpha = alpha, beta = beta, \
                            number_of_elements_power = number_of_elements_power, number_of_subdomains=number_of_subdomains, \
                            dof = dof, domain = domain, 
                            symmetry = symmetry, penalty = penalty, superpenalisation = superpenalisation,
                            max_iter = max_iter, linear_solver_tol = linear_solver_tol, dd_tol = dd_tol)
            

def eval_grid_search(n_eval = 10,\
                overlap = 1, alpha = 0.5, beta = 1, \
                number_of_elements_power = 4, number_of_subdomains = 4,\
                dof = 2, domain = [0,1], 
                symmetry = 1, penalty = 18, superpenalisation = False,
                max_iter = 50, linear_solver_tol = 1e-12, dd_tol = 1e-10):
    """ 
    Find best combination of alpha and beta
    """
    symmetry_params = [-1,0,1]
    
    alpha_list = np.linspace(0.1,2.1,n_eval)
    beta_list = np.linspace(0.1,2.1,n_eval)

    number_of_elements = 2**number_of_elements_power
    H0_all = []
    L2_all = []
    iterations = []
    residuals_all = []
    max_residuals = []
    

    # solve just with dg and without domain decomposition
    global_mesh = Mesh(cell_shape=(number_of_elements,number_of_elements), dof=dof, x_domain=domain, y_domain=domain)
    dg_global = DG(global_mesh, symmetry_parameter=symmetry, penalty=penalty, superpenalisation=superpenalisation, linear_solver_tol=linear_solver_tol, alpha=alpha, beta=0)
    dg_global.solve_poisson_matrix_free()
    dg_L2, dg_H0 = dg_global.compute_error_without_overlap()

    for beta in beta_list:
        for alpha in alpha_list:
            dd = DomainDecomposition(global_number_of_elements = number_of_elements, number_of_subdomains = number_of_subdomains,\
                                    overlap = overlap, dof = dof, domain = domain,  \
                                    alpha = alpha, beta=beta, \
                                    symmetry_parameter = symmetry, penalty = penalty, superpenalisation = superpenalisation, 
                                    linear_solver_tol = linear_solver_tol)
            i, H0, L2, residuals = dd.evaluate_additive_schwarz(max_iter = max_iter, max_residual = dd_tol)
            
            H0_all.append(H0)
            L2_all.append(L2)
            iterations.append(i)
            residuals_all.append(residuals)
            max_residuals.append(dd_tol)

        # make sure that beta has not a . anymore in the filename
        filename = f"beta_{10*beta}.pkl"
        fname_prefix = f"output/data/grid"
        pathname = fname_prefix + filename
        with open(pathname, 'wb') as f:
            pickle.dump((number_of_elements, number_of_subdomains,  iterations,\
                        H0_all, L2_all, residuals_all,\
                        overlap, alpha, beta,\
                        symmetry, penalty, superpenalisation,\
                        dg_L2, dg_H0, dd_tol)\
                        , f)
            
        plot_grid(pathname)