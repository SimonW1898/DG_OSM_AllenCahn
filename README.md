# DG_OSM_AllenCahn: Domain Decomposition with Discontinuous Galerkin Methods for the Allen-Cahn Equation

## Project Overview

This repository provides a modular Python implementation for combining Discontinuous Galerkin (DG) methods with domain decomposition techniques, including the Additive Schwarz Method (ASM) and Optimized Schwarz Method (OSM), to solve the nonlinear Allen-Cahn equation. The project enables numerical analysis of convergence properties and the impact of using Robin boundary conditions at subdomain interfaces.

**Highlights:**
- Class-based design for mesh generation, basis functions, DG solver, and domain decomposition.
- Numerical experiments on the Allen-Cahn equation, benchmarking pure DG versus DG + Schwarz methods.
- In-depth analysis of convergence properties and the impact of flux-based Robin boundary conditions.
- Clear code structure for extensibility and scientific experimentation.
- Comprehensive theoretical background and results documented in the included thesis PDF.

---

## Scientific Problem Tackled

The Allen-Cahn equation is a nonlinear reaction-diffusion PDE widely used in materials science to model phase separation in binary alloys. In this project, the equation is formulated as:

$$
\frac{1}{K}\frac{\partial u}{\partial t} - \Delta u + \frac{1}{\varepsilon^2} f(u) = 0 \quad \text{in} \; \Omega \times (0, T)
$$

with boundary and initial conditions:
- $u = 0$ on $\partial\Omega \times (0, T)$
- $u(x, 0) = u_0(x)$ in $\Omega$

where
- $f(u) = u(1-u)(1-2u) - \mu 6u(1-u)$
- $\Omega \subset \mathbb{R}^d$ is a bounded domain
- $K$, $\varepsilon$ are positive constants
- $u_0(x) = \frac{1}{2}\left(1 + \tanh\left(\frac{R_0 - x_1^2 - x_2^2}{2\varepsilon}\right)\right)$

The main objective is to combine DG methods with domain decomposition (Additive Schwarz and Optimized Schwarz) and analyze their performance, robustness, and convergence when solving the Allen-Cahn equation. Special emphasis is placed on the use of Robin boundary conditions at subdomain interfaces via OSM to improve convergence rates.

---

## Main Components

### 1. Discontinuous Galerkin (DG) Solver Modules

- **Class-Based Design:**  
  All numerical components (meshing, basis functions, assembly, solver routines) are implemented as Python classes for modularity and extensibility.
- **Flux-Based Formulation:**  
  DG's natural handling of discontinuities and fluxes is leveraged for flexible enforcement of boundary and interface conditions.
- **Nonlinear Reaction-Diffusion PDEs:**  
  Focused experiments on the Allen-Cahn equation, with extensibility to other PDEs.

### 2. Domain Decomposition Methods

- **Additive Schwarz Method (ASM):**  
  Overlapping subdomain approach, enabling efficient parallelization.
- **Optimized Schwarz Method (OSM):**  
  Robin boundary conditions are imposed at subdomain interfaces, exploiting DG's flux-based framework for improved convergence.
- **Iterative Solution:**  
  Subproblems are solved independently; global solution is obtained iteratively by exchanging interface data.

### 3. Analysis and Visualization

- **Convergence Studies:**  
  Numerical experiments evaluate the accuracy and convergence properties of the DG + domain decomposition approach, compared to pure DG methods.
- **Visualization:**  
  Solution profiles, interface behavior, and error metrics are visualized using Matplotlib.

---

## File Structure

```
DG_OSM_AllenCahn/
├── modules/                                
│   ├── __init__.py                          # Package initialization
│   ├── mesh.py                              # Mesh generation and handling
│   ├── submesh.py                           # Subdomain mesh handling for domain decomposition
│   ├── quadrature.py                        # Quadrature rules for numerical integration
│   ├── basis_functions.py                   # Discontinuous basis function definitions
│   ├── dg.py                                # Main DG solver implementation
│   ├── domain_decomposition.py              # Domain decomposition methods (ASM, OSM)
│   ├── dg_utils.py                          # Utility functions for evaluating DG convergence
│   └── domain_decomposition_utils.py        # Utility functions for domain decomposition performance
├── run_modules.py                           # Main entry point for running experiments
├── plot_radius.py                           # Evaluate the performance for varying radius
└── Thesis.pdf                               # Project documentation and theoretical background

```
---

## Dependencies

- **NumPy:** Numerical computations
- **SciPy:** Linear algebra and scientific routines
- **Matplotlib:** Visualization and plotting
- **Pandas:** Data handling and analysis

Install dependencies using:
```bash
pip install numpy scipy matplotlib pandas
```

---

## Usage

To run the main numerical experiments, execute:
```bash
python run_modules.py
```
All configuration and module calls are handled within the script; source code is fully modular for extension or further experimentation.

---

## Known Issues & Limitations

- The handling of Robin boundary conditions for non-overlapping subdomains in the Allen-Cahn equation remains an area for future improvement.
- Parallelization is not implemented out-of-the-box, but the code structure supports independent subdomain solves.

---

## Contributing

This project is intended for research and educational purposes. Contributions are welcome, especially for:
- Alternative interface conditions or Schwarz variants
- Multilevel domain decomposition
- Local mesh refinement
- Performance optimizations

---

## License

This project is for academic research. Please cite appropriately if used in publications.

---

## Author

**[Simon W](https://www.linkedin.com/in/simon-w-32183a292)**  

---

## Note

This repository is presented for professional review and demonstration of technical skills in scientific computing, numerical methods, and PDE solver design.  
For any questions, suggestions, or collaboration inquiries, feel free to contact me via LinkedIn or GitHub!
