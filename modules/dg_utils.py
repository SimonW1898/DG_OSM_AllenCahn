import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .mesh import Mesh
from.submesh import SubMesh
from .dg import DG


def create_dg_parameters(degree, penal, n, penal_list = []):
    """
    Create a dictionary with the parameters for the DG method
    """
    if penal_list == []:
        penal_list = np.zeros(degree)
        for i in range(degree):
            penal_list[i] = (i+1)*6
            
    dg_parameters = {
        "SIPG": {
            "symmetry": -1,
            "penal": penal,
            "degree": degree,
            "n": n,
            "N_quad": degree + 1
        },
        "IIPG": {
            "symmetry": 0, 
            "penal": penal,    
            "degree": degree,
            "n": n,
            "N_quad": degree + 1
        },
        "NIPG": {
            "symmetry": 1, 
            "penal": 1,    
            "degree": degree,
            "n": n,
            "N_quad": degree + 1
        },
        "NIPG0": {
            "symmetry": 1, 
            "penal": 0,    
            "degree": degree,
            "n": n,
            "N_quad": degree + 1
        }
    }
    return dg_parameters

def eval_dg_convergence(degree=2, max_refinement=4, superpenalization=1, alpha=1, beta=1):
    """   
    Evaluate the convergence of the DG method by increasing the number of elements
    """

    domain = [0, 1]
    methods = ["SIPG","IIPG","NIPG","NIPG0"]
    L2_error = np.zeros((len(methods), max_refinement))
    H0_error = np.zeros((len(methods), max_refinement))

    reference_line_L2 = np.zeros((len(methods), max_refinement))
    reference_line_H0 = np.zeros((len(methods), max_refinement))

    refinements = np.power(2, np.arange(3, max_refinement + 3))
    
    C_L2 = 1
    C_H0 = 1

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))


    data = []
    value = sum((i + 1) * 6 for i in range(degree))  # Calculate penal_val
    penal_val = value

    for i, selected_method in enumerate(methods):
        if superpenalization >= 3 or selected_method == "SIPG" or degree % 2 == 1:
            added_degree = 1
        else:
            added_degree = 0
        
        print(f"Method: {selected_method}")
        for j, n in enumerate(refinements):
            dg_parameters = create_dg_parameters(degree, penal_val, n)
            mesh = Mesh(cell_shape=(n, n), dof=degree, x_domain=domain, y_domain=domain)
            submesh = SubMesh(mesh, submesh_idx=0, overlap=1, number_of_submeshes=4)
            dg = DG(mesh=mesh,
                    penalty=dg_parameters[selected_method]["penal"], 
                    symmetry_parameter=dg_parameters[selected_method]["symmetry"], 
                    superpenalisation=superpenalization,
                    linear_solver_tol=1e-14,
                    alpha=alpha,
                    beta=beta)
            dg.solve_poisson_matrix_free()
            L2_error[i, j], H0_error[i, j] = dg.compute_error_without_overlap()

            # Calculate factor of the reference line
            if j == 1:
                C_L2 = dg.mesh.L2_error_without_overlap / dg.mesh.hx ** (degree +1 + added_degree)
                L2_rate = 0
                C_H0 = dg.mesh.H0_error_without_overlap / dg.mesh.hx ** (degree+1)
                H0_rate = 0

                data.append({
                    "Method": selected_method,
                    "Degree": degree,
                    "Superpenalization": superpenalization,
                    "Meshsize": refinements[j - 1],
                    "Penalty": dg_parameters[selected_method]["penal"],
                    "L2_Error": L2_error[i, j - 1],
                    "L2_Reference": C_L2 * (1 / refinements[j - 1]) ** (degree + added_degree),
                    "L2_rate": L2_rate,
                    "H0_Error": H0_error[i, j - 1],
                    "H0_Reference": C_H0 * (1 / refinements[j - 1]) ** (degree),
                    "H0_rate": H0_rate    
                })
                reference_line_L2[i, j - 1] = C_L2 * (1 / refinements[j - 1]) ** (degree + added_degree)
                reference_line_H0[i, j - 1] = C_H0 * (1 / refinements[j - 1]) ** (degree)

            if j >= 1:
                L2_rate = (np.log(L2_error[i, j - 1]) - np.log(L2_error[i, j])) / (np.log(1 / refinements[j - 1]) - np.log(1 / refinements[j]))
                H0_rate = (np.log(H0_error[i, j - 1]) - np.log(H0_error[i, j])) / (np.log(1 / refinements[j - 1]) - np.log(1 / refinements[j]))

                data.append({
                    "Method": selected_method,
                    "Degree": degree,
                    "Meshsize": n,
                    "Penalty": dg_parameters[selected_method]["penal"],
                    "L2_Error": L2_error[i, j],
                    "L2_Reference": C_L2 * dg.mesh.hx ** (degree + added_degree),
                    "L2_rate": L2_rate,
                    "H0_Error": H0_error[i, j],
                    "H0_Reference": C_H0 * (1 / refinements[j - 1]) ** (degree),
                    "H0_rate": H0_rate
                })
                
                reference_line_L2[i, j] = C_L2 * dg.mesh.hx ** (degree +1 + added_degree)
                reference_line_H0[i, j] = C_H0 * dg.mesh.hx ** (degree+1)

        # Plot L2 error
        color = ax1._get_lines.get_next_color()
        
        ax1.loglog(refinements * refinements, L2_error[i, :], "-o", label=selected_method, color=color)
        ax2.loglog(refinements * refinements, H0_error[i, :], "-o", label=selected_method, color=color)

        color_H0 = ax2._get_lines.get_next_color()
        ax1.loglog(refinements * refinements, reference_line_L2[i, :], "--", color=color)
        ax2.loglog(refinements * refinements, reference_line_H0[i, :], "--", color=color)

    # Configure L2 plot
    ax1.semilogy([], [], "--", label="reference lines", color = color)
    ax1.set_title("Convergence rate increasing the refinement level (L2 Error)")
    ax1.set_xlabel("Number of cells")
    ax1.set_ylabel("L2 norm")
    ax1.legend()


    # Configure H0 plot
    ax2.semilogy([], [], "--", label="reference lines", color = color)
    ax2.set_title("Convergence rate increasing the refinement level (H0 Error)")
    ax2.set_xlabel("Number of cells")
    ax2.set_ylabel("H0 norm")
    ax2.legend()

    # do the suptitle with the parameters alpha and beta
    fig.suptitle(f"Convergence rate increasing the refinement level (Degree {degree}, Superpenalization {superpenalization}, Alpha {alpha}, Beta {beta})")

    plt.tight_layout()  # Adjusts the layout for better fit
    plt.show()

    # save the plot
    fig.savefig("output/plots/convergence_DG.png")
    print("Plot saved in output/plots/convergence_DG.png")
    df = pd.DataFrame(data)
    print(df)
    # store the data in a csv file
    df.to_csv("output/data/convergence_DG.csv", index=False)

    # print all solution for the last dg
    dg.mesh.plot_all_solutions(subdomains=0, overlap=0, alpha=alpha, beta=beta, symmetry= 0)

    return df





if __name__ == "__main__":
    df = eval_dg_convergence(degree = 2, max_refinement = 6, superpenalization = 1)
    df.to_csv("convergence_rate.csv", index = False)
    print("successful run dg_utils")