"""
COBYLA Optimizer for VQE
Optimizes parameterized quantum circuits for molecular ground state energy
"""

import numpy as np
import time
from typing import Tuple, List, Callable
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import StatevectorSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class VQEOptimizer:
    """
    COBYLA-based VQE optimizer with detailed progress tracking.
    """
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6):
        """
        Initialize the VQE optimizer.
        
        Args:
            max_iterations: Maximum COBYLA iterations
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.backend = StatevectorSimulator()
        
        # Tracking variables
        self.energy_history = []
        self.evaluation_count = 0
        self.best_energy = float('inf')
        self.best_parameters = None
        
        print(f"VQE Optimizer initialized:")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Backend: {self.backend.name}")
    
    def create_cost_function(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Callable:
        """
        Create the cost function for VQE optimization.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian
            
        Returns:
            Cost function for optimization
        """
        # Create estimator
        estimator = BackendEstimatorV2(backend=self.backend)
        
        # Transpile ansatz for efficiency
        pass_manager = generate_preset_pass_manager(
            optimization_level=2,
            backend=self.backend
        )
        transpiled_ansatz = pass_manager.run(ansatz)
        
        def cost_function(parameters: np.ndarray) -> float:
            """Evaluate energy for given parameters."""
            eval_start_time = time.time()
            self.evaluation_count += 1
            
            try:
                # Bind parameters
                bound_circuit = transpiled_ansatz.assign_parameters(parameters)
                
                # Run estimation
                pub = (bound_circuit, hamiltonian)
                job = estimator.run([pub])
                result = job.result()
                
                # Extract energy
                if hasattr(result[0].data, 'evs'):
                    energy = float(result[0].data.evs)
                else:
                    energy = float(result[0].data.expectation_values)
                
                # Track progress
                self.energy_history.append(energy)
                eval_time = time.time() - eval_start_time
                
                # Update best result
                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_parameters = parameters.copy()
                
                # Print progress with timing
                print(f"  Evaluation {self.evaluation_count:2d}: Energy = {energy:.8f} Hartree (took {eval_time:.2f}s)")
                
                # Highlight new best energy
                if energy == self.best_energy:
                    print(f"    *** NEW BEST ENERGY: {energy:.8f} Hartree ***")
                
                return energy
                
            except Exception as e:
                print(f"    WARNING: Evaluation {self.evaluation_count} failed: {e}")
                return 1e10
        
        return cost_function
    
    def optimize_vqe(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Tuple[float, np.ndarray, List[float]]:
        """
        Run VQE optimization using COBYLA.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian
            
        Returns:
            Tuple of (optimal_energy, optimal_parameters, energy_history)
        """
        print("=" * 60)
        print("VQE OPTIMIZATION")
        print("=" * 60)
        
        # Reset tracking
        self.energy_history = []
        self.evaluation_count = 0
        self.best_energy = float('inf')
        self.best_parameters = None
        
        # Create cost function
        cost_function = self.create_cost_function(ansatz, hamiltonian)
        
        # Initialize parameters
        num_parameters = ansatz.num_parameters
        initial_params = np.random.normal(0, 0.1, num_parameters)
        
        # Set parameter bounds for rotation gates
        bounds = [(0, 2*np.pi) for _ in range(num_parameters)]
        
        # Convert bounds to constraints for COBYLA
        constraints = []
        for i, (lower, upper) in enumerate(bounds):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i, low=lower: x[idx] - low
            })
            constraints.append({
                'type': 'ineq', 
                'fun': lambda x, idx=i, up=upper: up - x[idx]
            })
        
        print(f"Starting COBYLA optimization...")
        print(f"  Parameters: {num_parameters}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Parameter bounds: [0, 2π] for all rotation gates")
        print("=" * 60)
        
        optimization_start_time = time.time()
        
        # Run optimization
        print("STARTING FUNCTION EVALUATIONS:")
        result = minimize(
            cost_function,
            initial_params,
            method='cobyla',
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'tol': self.tolerance,
                'disp': True
            }
        )
        
        optimization_end_time = time.time()
        total_optimization_time = optimization_end_time - optimization_start_time
        
        # Use best result found during optimization
        final_energy = self.best_energy if self.best_energy < 1e9 else result.fun
        final_params = self.best_parameters if self.best_parameters is not None else result.x
        
        # Display summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Final energy: {final_energy:.8f} Hartree")
        print(f"COBYLA success: {result.success}")
        print(f"COBYLA message: {result.message}")
        print(f"Total function evaluations: {result.nfev}")
        print(f"Total optimization time: {total_optimization_time:.2f} seconds")
        print(f"Average time per evaluation: {total_optimization_time/result.nfev:.2f} seconds")
        
        if self.energy_history:
            initial_energy = self.energy_history[0]
            energy_improvement = initial_energy - final_energy
            print(f"Energy improvement: {energy_improvement:.8f} Hartree")
        
        print("=" * 60)
        
        return final_energy, final_params, self.energy_history.copy()


def optimize_vqe(ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, max_iterations: int = 10) -> Tuple[float, np.ndarray, List[float]]:
    """
    Main function: Optimize VQE using COBYLA.
    
    Args:
        ansatz: Parameterized quantum circuit
        hamiltonian: Molecular Hamiltonian
        max_iterations: Maximum optimization iterations
        
    Returns:
        Tuple of (optimal_energy, optimal_parameters, energy_history)
    """
    optimizer = VQEOptimizer(max_iterations=max_iterations, tolerance=1e-6)
    return optimizer.optimize_vqe(ansatz, hamiltonian)


if __name__ == "__main__":
    # Test optimizer with a simple system
    from qiskit.circuit import ParameterVector
    
    print("Testing VQE optimizer...")
    
    # Create simple test ansatz
    num_qubits = 4
    ansatz = QuantumCircuit(num_qubits)
    params = ParameterVector('theta', 8)
    
    for i in range(num_qubits):
        ansatz.ry(params[i], i)
    for i in range(num_qubits):
        ansatz.rz(params[i+4], i)
    
    # Simple test Hamiltonian
    hamiltonian = SparsePauliOp.from_list([
        ("ZIII", -1.0),
        ("IZII", -0.5),
        ("IIZI", -0.5),
        ("IIIZ", -1.0),
        ("ZZII", 0.25),
    ])
    
    print(f"Test system: {num_qubits} qubits, {ansatz.num_parameters} parameters")
    
    # Run optimization
    energy, params, history = optimize_vqe(ansatz, hamiltonian, max_iterations=5)
    
    print(f"Test completed. Final energy: {energy:.6f} Hartree")
