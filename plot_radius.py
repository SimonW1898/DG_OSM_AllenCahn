import matplotlib.pyplot as plt
import numpy as np

def plot_radius_from_file(file_path):
    """
    Reads radius data from a file and plots it.
    :param file_path: Path to the file containing radius data
    """
    time_steps = []
    radius_analytical = []
    radius_numerical = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        metadata = {line.split(": ")[0][2:]: line.split(": ")[1].strip() for line in lines if line.startswith("#") and ":" in line}
        
        for line in lines:
            if not line.startswith("#"):
                values = line.strip().split()
                if len(values) == 3:
                    time_steps.append(float(values[0]))
                    radius_analytical.append(float(values[1]))
                    radius_numerical.append(float(values[2]))
    
    # Plot the radius over time
    plt.figure()
    plt.plot(time_steps, radius_analytical, '-', label="Analytical")
    plt.plot(time_steps, radius_numerical, '-', label="Numerical")
    plt.legend()
    plt.title(f"Radius over time\n Time Resolution: {metadata.get('time_resolution', 'Unknown')}, Meshsize: {metadata.get('mesh_size', 'Unknown')}")
    plt.xlabel("Time")
    plt.ylabel("Radius")
    plt.tight_layout()
    
    output_dir = metadata.get("output_directory", "./")
    output_path = f"{output_dir}/radius_over_time.pdf"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Radius plot saved at {output_path}")

def plot_radius_from_file_dd(file_path):
    """
    Reads radius data from a file and plots analytical, DG, and DD radii.
    
    :param file_path: Path to the file containing radius data.
    """
    time_steps = []
    radius_analytical = []
    radius_numerical_dd = []
    radius_numerical_dg = []
    metadata = {}

    # Read file and extract metadata
    with open(file_path, "r") as f:
        lines = f.readlines()
        metadata = {line.split(": ")[0][2:]: line.split(": ")[1].strip() for line in lines if line.startswith("#") and ":" in line}

        for line in lines:
            if not line.startswith("#"):
                values = line.strip().split()
                if len(values) == 4:  # Ensure the line has all four values
                    time_steps.append(float(values[0]))
                    radius_analytical.append(float(values[1]))
                    radius_numerical_dd.append(float(values[2]))
                    radius_numerical_dg.append(float(values[3]))

    # Extract metadata values with defaults
    time_resolution = metadata.get("time_resolution", "Unknown")
    mesh_size = metadata.get("mesh_size", "Unknown")
    overlap = metadata.get("overlap", "Unknown")
    stability_param = metadata.get("stability_parameter", "Unknown")
    symmetry_param = metadata.get("symmetry_parameter", "Unknown")

    # Plot the radius over time
    plt.figure()
    plt.plot(time_steps, radius_analytical, '-', label="Analytical")
    plt.plot(time_steps, radius_numerical_dd, '--', label="Numerical DD (Additive Schwarz)")
    plt.plot(time_steps, radius_numerical_dg, ':', label="Numerical DG (Pure)")
    plt.legend()
    
    # Title with metadata information
    plt.title(f"Radius over time\nTime Res: {time_resolution}, Mesh: {mesh_size}, "
              f"Overlap: {overlap},\n Stability: {stability_param}, Symmetry: {symmetry_param}")
    
    plt.xlabel("Time")
    plt.ylabel("Radius")
    plt.tight_layout()

    # Save the figure
    output_dir = metadata.get("output_directory", "./")
    output_path = f"{output_dir}/radius_over_time.pdf"
    plt.savefig(output_path)
    plt.close()

    print(f"Radius plot saved at {output_path}")



if __name__ == "__main__":
    file_path = "radius.txt"
    plot_radius_from_file_dd(file_path)
