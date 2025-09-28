"""
VQE Molecular Simulator - Modular Implementation
Integrates Hamiltonian generation, ansatz creation, and optimization for molecular VQE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import custom modules
try:
    from hamiltonian import generate_hamiltonian
    HAMILTONIAN_AVAILABLE = True
except ImportError:
    print("Error: hamiltonian.py not found. Please ensure it's in the same directory.")
    HAMILTONIAN_AVAILABLE = False

try:
    from ansatz import create_ansatz
    ANSATZ_AVAILABLE = True
except ImportError:
    print("Error: ansatz.py not found. Please ensure it's in the same directory.")
    ANSATZ_AVAILABLE = False

try:
    from optimizer import optimize_vqe
    OPTIMIZER_AVAILABLE = True
except ImportError:
    print("Error: optimizer.py not found. Please ensure it's in the same directory.")
    OPTIMIZER_AVAILABLE = False


@dataclass
class VQEResult:
    """Container for VQE simulation results."""
    ground_state_energy: float
    optimal_parameters: np.ndarray
    energy_history: List[float]
    evaluation_count: int
    execution_time: float
    molecular_formula: str
    num_qubits: int
    num_parameters: int
    success: bool


class MolecularVQEPipeline:
    """
    Complete VQE simulation pipeline integrating all modules.
    """
    
    def __init__(self, max_iterations: int = 10, ansatz_reps: int = 2):
        """
        Initialize the VQE pipeline.
        
        Args:
            max_iterations: Maximum COBYLA iterations
            ansatz_reps: Number of ansatz repetition layers
        """
        self.max_iterations = max_iterations
        self.ansatz_reps = ansatz_reps
        
        # Tracking variables
        self.start_time = None
        
        print("Molecular VQE Pipeline Initialized")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Ansatz repetitions: {ansatz_reps}")
    
    def run_simulation(self, csv_path: str) -> VQEResult:
        # Run complete VQE simulation from CSV file.

        print("=" * 70)
        print("MOLECULAR VQE SIMULATION")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Step 1: Generate Hamiltonian
        print("Step 1: Generating molecular Hamiltonian...")
        hamiltonian, properties = generate_hamiltonian(csv_path)
        
        # Step 2: Create Ansatz
        print("\nStep 2: Creating quantum ansatz...")
        num_qubits = properties["num_qubits"]
        num_electrons = properties["active_space_electrons"]
        ansatz = create_ansatz(num_qubits, num_electrons, self.ansatz_reps)
        
        # Step 3: VQE Optimization
        print("\nStep 3: Running VQE optimization...")
        optimal_energy, optimal_params, energy_history = optimize_vqe(
            ansatz, hamiltonian, self.max_iterations
        )
        
        execution_time = time.time() - self.start_time
        
        # Create molecular formula string
        composition = properties['composition']
        formula_parts = []
        for atom, count in sorted(composition.items()):
            if count == 1:
                formula_parts.append(atom)
            else:
                formula_parts.append(f"{atom}{count}")
        molecular_formula = "".join(formula_parts)
        
        # Create result object
        result = VQEResult(
            ground_state_energy=optimal_energy,
            optimal_parameters=optimal_params,
            energy_history=energy_history,
            evaluation_count=len(energy_history),
            execution_time=execution_time,
            molecular_formula=molecular_formula,
            num_qubits=num_qubits,
            num_parameters=len(optimal_params),
            success=optimal_energy < 1e9
        )
        
        # Generate output files
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        
        # Save energy convergence CSV
        self.save_energy_convergence_csv(base_filename, energy_history)
        
        # Create visualization
        self.create_energy_convergence_plot(result, base_filename)
        
        # Display summary
        print("\n" + "=" * 70)
        print("VQE SIMULATION COMPLETED")
        print("=" * 70)
        print(f"Molecular formula: {molecular_formula}")
        print(f"Ground state energy: {optimal_energy:.8f} Hartree")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Function evaluations: {len(energy_history)}")
        print(f"Success: {result.success}")
        print("=" * 70)
        
        return result
    
    def save_energy_convergence_csv(self, base_filename: str, energy_history: List[float]):
        """
        Save energy convergence data to CSV file.
        
        Args:
            base_filename: Base filename for output
            energy_history: List of energy values
        """
        output_filename = f"{base_filename}_energy.csv"
        
        # Create DataFrame with exact format requested
        df = pd.DataFrame({
            'Function Evaluation': range(1, len(energy_history) + 1),
            'Energy (Hartree)': energy_history
        })
        
        # Save to CSV
        df.to_csv(output_filename, index=False)
        print(f"Energy convergence saved to: {output_filename}")
    
    def create_energy_convergence_plot(self, result: VQEResult, base_filename: str):
        # Create energy convergence visualization.

        plt.figure(figsize=(10, 6))
        
        # Plot energy convergence
        evaluations = range(1, len(result.energy_history) + 1)
        plt.plot(evaluations, result.energy_history, 'b-o', linewidth=2, markersize=4)
        
        # Add horizontal line for final energy
        plt.axhline(y=result.ground_state_energy, color='r', linestyle='--', 
                   label=f'Final Energy: {result.ground_state_energy:.6f} Hartree')
        
        # Formatting
        plt.xlabel('Function Evaluation', fontsize=12)
        plt.ylabel('Energy (Hartree)', fontsize=12)
        plt.title(f'VQE Energy Convergence\n{result.molecular_formula}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with system info
        info_text = f"""System Info:
Qubits: {result.num_qubits}
Parameters: {result.num_parameters}
Evaluations: {result.evaluation_count}
Time: {result.execution_time:.1f}s"""
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        output_filename = f"{base_filename}_vqe_results.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Energy convergence plot saved to: {output_filename}")
        
        plt.close()


def main():
    """Main function to run VQE simulation from command line."""
    if len(sys.argv) != 2:
        print("Usage: python vqe.py <csv_file_path>")
        print("Example: python vqe.py molecule.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found")
        print("Please check the file path and try again.")
        sys.exit(1)
    
    # Check if all required modules are available
    if not all([HAMILTONIAN_AVAILABLE, ANSATZ_AVAILABLE, OPTIMIZER_AVAILABLE]):
        print("Error: Required modules not found.")
        print("Please ensure hamiltonian.py, ansatz.py, and optimizer.py are in the same directory.")
        sys.exit(1)
    
    try:
        # Create VQE pipeline
        pipeline = MolecularVQEPipeline(
            max_iterations=10,
            ansatz_reps=2
        )
        
        # Run complete simulation
        result = pipeline.run_simulation(csv_path)
        
        # Show output files created
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\nOutput files created:")
        print(f"  - {base_filename}_energy.csv (energy convergence data)")
        print(f"  - {base_filename}_vqe_results.png (energy convergence plot)")
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()