"""
COBYLA-Only Optimizer for Molecular VQE
Streamlined implementation focused on robustness for quantum chemistry applications
Optimized for drug molecules like amoxicillin with noisy quantum cost landscapes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import time
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

# Use correct V2 primitives import
try:
    from qiskit_aer.primitives import EstimatorV2 as Estimator
    ESTIMATOR_V2_AVAILABLE = True
except ImportError:
    try:
        from qiskit_aer.primitives import Estimator
        ESTIMATOR_V2_AVAILABLE = False
        print("Warning: Using V1 Estimator. Consider upgrading to Qiskit 1.0+ for V2 primitives.")
    except ImportError:
        from qiskit.primitives import Estimator
        ESTIMATOR_V2_AVAILABLE = False
        print("Warning: Using reference Estimator implementation.")


@dataclass
class COBYLAResult:
    """
    Optimized result container for COBYLA-only optimization.
    """
    optimal_parameters: np.ndarray
    optimal_energy: float
    scipy_result: OptimizeResult
    energy_history: List[float]
    parameter_history: List[np.ndarray]
    evaluation_times: List[float]
    total_evaluations: int
    total_time: float
    convergence_info: Dict[str, Any]
    success: bool


class COBYLAMolecularOptimizer:
    """
    COBYLA-only optimizer for molecular VQE calculations.
    
    Designed specifically for robustness on noisy quantum cost landscapes
    with minimal computational overhead and maximum reliability.
    """
    
    def __init__(self, 
                 backend: Optional[Any] = None,
                 shots: int = 2048,
                 optimization_level: int = 2,
                 seed: Optional[int] = None,
                 max_iterations: int = 500,
                 tolerance: float = 1e-6):
        """
        Initialize the COBYLA-only VQE optimizer.
        
        Args:
            backend: Quantum backend (uses AerSimulator if None)
            shots: Number of shots for quantum measurements
            optimization_level: Circuit optimization level (0-3)
            seed: Random seed for reproducibility
            max_iterations: Maximum COBYLA iterations
            tolerance: COBYLA convergence tolerance
        """
        # Backend setup
        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend
            
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed = seed
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize transpiler
        self.pass_manager = generate_preset_pass_manager(
            optimization_level=self.optimization_level,
            backend=self.backend
        )
        
        # Tracking variables
        self.energy_history = []
        self.parameter_history = []
        self.evaluation_times = []
        self.evaluation_count = 0
        self.start_time = None
        
        # State tracking
        self.best_energy = float('inf')
        self.best_parameters = None
        
        estimator_version = "V2" if ESTIMATOR_V2_AVAILABLE else "V1"
        print(f"COBYLA Molecular Optimizer initialized:")
        print(f"  Backend: {self.backend}")
        print(f"  Shots: {self.shots}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Tolerance: {self.tolerance}")
        print(f"  Estimator version: {estimator_version}")
        
    def create_cost_function(self, 
                           ansatz: QuantumCircuit, 
                           hamiltonian: SparsePauliOp) -> Callable:
        """
        Create the cost function for VQE optimization.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian as SparsePauliOp
            
        Returns:
            Cost function optimized for COBYLA
        """
        # Create estimator
        if ESTIMATOR_V2_AVAILABLE:
            run_options = {'shots': self.shots}
            if self.seed is not None:
                run_options['seed_simulator'] = self.seed
                
            options = {
                'run_options': run_options,
                'backend_options': {}
            }
            estimator = Estimator(options=options)
        else:
            estimator = Estimator(backend=self.backend, options={'shots': self.shots})
        
        # Transpile ansatz once for efficiency
        transpiled_ansatz = self.pass_manager.run(ansatz)
        
        def cost_function(parameters: np.ndarray) -> float:
            """
            Evaluate energy for given parameters.
            Optimized for minimal overhead and robust error handling.
            """
            eval_start = time.time()
            self.evaluation_count += 1
            
            try:
                # Bind parameters
                bound_circuit = transpiled_ansatz.assign_parameters(parameters)
                
                # Run estimation with version-specific handling
                if ESTIMATOR_V2_AVAILABLE:
                    pub = (bound_circuit, hamiltonian)
                    job = estimator.run([pub])
                    result = job.result()
                    
                    # Extract energy value
                    pub_result = result[0]
                    if hasattr(pub_result.data, 'evs'):
                        energy = float(pub_result.data.evs)
                    else:
                        energy = float(pub_result.data.expectation_values)
                else:
                    job = estimator.run([bound_circuit], [hamiltonian])
                    result = job.result()
                    energy = float(result.values[0])
                
                # Track progress
                eval_time = time.time() - eval_start
                self.energy_history.append(energy)
                self.parameter_history.append(parameters.copy())
                self.evaluation_times.append(eval_time)
                
                # Update best result
                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_parameters = parameters.copy()
                
                # Progress reporting (less frequent for COBYLA efficiency)
                if self.evaluation_count % 20 == 0:
                    print(f"  Evaluation {self.evaluation_count}: Energy = {energy:.8f} Hartree")
                
                return energy
                
            except Exception as e:
                print(f"Warning: Evaluation {self.evaluation_count} failed: {e}")
                # Return large penalty instead of inf to help COBYLA
                return 1e10
        
        return cost_function
    
    def initialize_parameters(self, 
                            num_parameters: int,
                            strategy: str = "random_small") -> np.ndarray:
        """
        Initialize parameters with strategies optimized for COBYLA.
        
        Args:
            num_parameters: Number of parameters
            strategy: Initialization strategy
            
        Returns:
            Initial parameter array
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if strategy == "zero":
            return np.zeros(num_parameters)
        
        elif strategy == "random_small":
            # Conservative small random values - good for COBYLA
            return np.random.normal(0, 0.05, num_parameters)
        
        elif strategy == "random_bounded":
            # Random within [0, π] - reasonable for rotation gates
            return np.random.uniform(0, np.pi, num_parameters)
        
        elif strategy == "layered_conservative":
            # Layer-aware initialization with conservative scaling
            params = np.zeros(num_parameters)
            # Assume 4 layers for typical EfficientSU2
            layer_size = max(1, num_parameters // 4)
            
            for i in range(num_parameters):
                layer = i // layer_size
                # Very conservative scaling to avoid large initial parameter space
                scale = 0.01 * (1 + 0.1 * layer)
                params[i] = np.random.normal(0, scale)
            
            return params
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
    
    def optimize(self,
                ansatz: QuantumCircuit,
                hamiltonian: SparsePauliOp,
                initial_strategy: str = "random_small",
                bounds: Optional[List[Tuple[float, float]]] = None,
                retry_on_failure: bool = True) -> COBYLAResult:
        """
        Run COBYLA optimization with optional retry strategies.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian
            initial_strategy: Parameter initialization strategy
            bounds: Parameter bounds for constraints
            retry_on_failure: Whether to retry with different initialization
            
        Returns:
            COBYLA optimization result
        """
        print("=" * 60)
        print("COBYLA MOLECULAR VQE OPTIMIZATION")
        print("=" * 60)
        
        self.start_time = time.time()
        num_parameters = ansatz.num_parameters
        
        # Reset tracking
        self.energy_history = []
        self.parameter_history = []
        self.evaluation_times = []
        self.evaluation_count = 0
        self.best_energy = float('inf')
        self.best_parameters = None
        
        # Create cost function
        cost_function = self.create_cost_function(ansatz, hamiltonian)
        
        print(f"System: {ansatz.num_qubits} qubits, {num_parameters} parameters")
        print(f"Hamiltonian: {len(hamiltonian)} terms")
        
        # Initialize parameters
        initial_params = self.initialize_parameters(num_parameters, initial_strategy)
        
        # Run COBYLA optimization
        scipy_result = self._run_cobyla(cost_function, initial_params, bounds)
        
        # Check if retry is needed and beneficial
        if retry_on_failure and not scipy_result.success and self.evaluation_count < self.max_iterations:
            print("\nFirst optimization unsuccessful. Trying alternative initialization...")
            
            # Try different initialization strategy
            retry_strategy = "layered_conservative" if initial_strategy == "random_small" else "random_bounded"
            retry_params = self.initialize_parameters(num_parameters, retry_strategy)
            
            # Adjust remaining iterations
            remaining_iters = self.max_iterations - self.evaluation_count
            original_max_iter = self.max_iterations
            self.max_iterations = remaining_iters
            
            retry_result = self._run_cobyla(cost_function, retry_params, bounds)
            
            # Use better result
            if retry_result.fun < scipy_result.fun:
                scipy_result = retry_result
                print(f"Retry successful: improved energy to {retry_result.fun:.8f}")
            
            # Restore original max_iterations
            self.max_iterations = original_max_iter
        
        # Create final result
        result = self._create_result(scipy_result, initial_strategy)
        
        # Display summary
        print("\n" + "=" * 60)
        print("COBYLA OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Final Energy: {result.optimal_energy:.8f} Hartree")
        print(f"Total Evaluations: {result.total_evaluations}")
        print(f"Total Time: {result.total_time:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Average time per evaluation: {result.total_time/result.total_evaluations:.3f} seconds")
        print("=" * 60)
        
        return result
    
    def _run_cobyla(self,
                   cost_function: Callable,
                   initial_parameters: np.ndarray,
                   bounds: Optional[List[Tuple[float, float]]]) -> OptimizeResult:
        """Run a single COBYLA optimization."""
        
        print(f"\nStarting COBYLA optimization...")
        print(f"  Parameters: {len(initial_parameters)}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Tolerance: {self.tolerance}")
        
        # Set up constraints for bounds
        constraints = []
        if bounds is not None:
            for i, (lower, upper) in enumerate(bounds):
                # Lower bound constraint
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=i, low=lower: x[idx] - low
                })
                # Upper bound constraint
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda x, idx=i, up=upper: up - x[idx]
                })
        
        # COBYLA options optimized for molecular systems
        options = {
            'maxiter': self.max_iterations,
            'tol': self.tolerance,
            'disp': True,
            'catol': 1e-8  # Constraint tolerance
        }
        
        # Run optimization
        result = minimize(
            cost_function,
            initial_parameters,
            method='cobyla',
            constraints=constraints,
            options=options
        )
        
        print(f"COBYLA completed: Energy = {result.fun:.8f}")
        return result
    
    def _create_result(self, 
                      scipy_result: OptimizeResult,
                      strategy_used: str) -> COBYLAResult:
        """Create comprehensive result object."""
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Convergence analysis
        convergence_info = {
            'strategy_used': strategy_used,
            'scipy_success': scipy_result.success,
            'final_energy_variance': np.var(self.energy_history[-10:]) if len(self.energy_history) >= 10 else None,
            'energy_improvement': self.energy_history[0] - self.best_energy if self.energy_history else 0,
            'avg_evaluation_time': np.mean(self.evaluation_times) if self.evaluation_times else 0
        }
        
        # Determine overall success
        success = (
            scipy_result.success and 
            self.best_energy < 1e9 and  # Exclude penalty values
            len(self.energy_history) > 5  # Minimum evaluations
        )
        
        return COBYLAResult(
            optimal_parameters=self.best_parameters if self.best_parameters is not None else scipy_result.x,
            optimal_energy=self.best_energy if self.best_energy < 1e9 else scipy_result.fun,
            scipy_result=scipy_result,
            energy_history=self.energy_history.copy(),
            parameter_history=self.parameter_history.copy(),
            evaluation_times=self.evaluation_times.copy(),
            total_evaluations=self.evaluation_count,
            total_time=total_time,
            convergence_info=convergence_info,
            success=success
        )
    
    def visualize_optimization(self, 
                             result: COBYLAResult,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        Create visualization focused on COBYLA-specific metrics.
        
        Args:
            result: COBYLA optimization result
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('COBYLA VQE Optimization Analysis', fontsize=16, fontweight='bold')
        
        # Energy convergence
        axes[0, 0].plot(result.energy_history, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].axhline(y=result.optimal_energy, color='r', linestyle='--', 
                          label=f'Best: {result.optimal_energy:.6f}')
        axes[0, 0].set_xlabel('Function Evaluation')
        axes[0, 0].set_ylabel('Energy (Hartree)')
        axes[0, 0].set_title('Energy Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy improvement over time
        if len(result.energy_history) > 1:
            energy_array = np.array(result.energy_history)
            running_best = np.minimum.accumulate(energy_array)
            axes[0, 1].plot(running_best, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Function Evaluation')
            axes[0, 1].set_ylabel('Best Energy So Far (Hartree)')
            axes[0, 1].set_title('Best Energy Progress')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter evolution (sample of parameters)
        if result.parameter_history:
            param_array = np.array(result.parameter_history)
            num_params_to_plot = min(6, param_array.shape[1])
            
            for i in range(num_params_to_plot):
                axes[1, 0].plot(param_array[:, i], label=f'θ_{i}', alpha=0.7)
            
            axes[1, 0].set_xlabel('Function Evaluation')
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].set_title('Parameter Evolution (first 6)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation timing
        if result.evaluation_times:
            axes[1, 1].plot(result.evaluation_times, 'purple', alpha=0.7)
            axes[1, 1].axhline(y=np.mean(result.evaluation_times), color='orange', 
                              linestyle='--', label=f'Avg: {np.mean(result.evaluation_times):.3f}s')
            axes[1, 1].set_xlabel('Function Evaluation')
            axes[1, 1].set_ylabel('Evaluation Time (s)')
            axes[1, 1].set_title('Timing Analysis')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = f"""
        Total Evaluations: {result.total_evaluations}
        Total Time: {result.total_time:.2f}s
        Success: {result.success}
        Strategy: {result.convergence_info['strategy_used']}
        Final Energy: {result.optimal_energy:.8f}
        Energy Improvement: {result.convergence_info['energy_improvement']:.6f}
        """
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Optimization plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig


def optimize_molecular_cobyla(ansatz: QuantumCircuit,
                             hamiltonian: SparsePauliOp,
                             config: Optional[Dict] = None) -> COBYLAResult:
    """
    High-level function for COBYLA-only molecular VQE optimization.
    
    Args:
        ansatz: Parameterized quantum circuit
        hamiltonian: Molecular Hamiltonian
        config: Configuration dictionary
        
    Returns:
        COBYLA optimization result
    """
    # Default configuration optimized for molecular systems
    default_config = {
        'shots': 2048,
        'max_iterations': 500,
        'tolerance': 1e-6,
        'seed': 42,
        'initial_strategy': 'random_small',
        'bounds': None,
        'retry_on_failure': True
    }
    
    if config is not None:
        default_config.update(config)
    
    # Auto-detect bounds for rotation gates
    if default_config['bounds'] is None:
        num_params = ansatz.num_parameters
        default_config['bounds'] = [(0, 2*np.pi) for _ in range(num_params)]
    
    # Create optimizer
    optimizer = COBYLAMolecularOptimizer(
        shots=default_config['shots'],
        max_iterations=default_config['max_iterations'],
        tolerance=default_config['tolerance'],
        seed=default_config['seed']
    )
    
    # Run optimization
    result = optimizer.optimize(
        ansatz,
        hamiltonian,
        initial_strategy=default_config['initial_strategy'],
        bounds=default_config['bounds'],
        retry_on_failure=default_config['retry_on_failure']
    )
    
    return result


if __name__ == "__main__":
    # Test with simple system
    print("COBYLA Molecular Optimizer Test")
    print("=" * 40)
    
    from qiskit.circuit.library import efficient_su2
    from qiskit.quantum_info import SparsePauliOp
    
    # Create test system
    num_qubits = 4
    ansatz = efficient_su2(num_qubits, reps=2)
    
    # Simple test Hamiltonian
    hamiltonian = SparsePauliOp.from_list([
        ("ZIII", -1.0),
        ("IZII", -0.5),
        ("IIZI", -0.5),
        ("IIIZ", -1.0),
        ("ZZII", 0.25),
        ("ZIZI", 0.25)
    ])
    
    print(f"Test system: {num_qubits} qubits, {ansatz.num_parameters} parameters")
    
    # Run COBYLA optimization
    config = {
        'max_iterations': 100,
        'shots': 1024,
        'tolerance': 1e-6,
        'seed': 42
    }
    
    result = optimize_molecular_cobyla(ansatz, hamiltonian, config)
    
    # Create visualization
    optimizer = COBYLAMolecularOptimizer()
    fig = optimizer.visualize_optimization(result, save_path='cobyla_test.png')
    
    print("COBYLA test completed successfully!")