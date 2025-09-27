"""
EfficientSU2 Ansatz Implementation for Molecular VQE
Specifically optimized for amoxicillin and similar large drug molecules

Based on IBM Quantum EfficientSU2 documentation:
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.EfficientSU2
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import warnings

# Qiskit imports - using the function-based API to avoid deprecation
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp


class EfficientSU2MolecularAnsatz:
    """
    EfficientSU2 ansatz specifically designed for molecular VQE calculations.
    
    This implementation focuses on the EfficientSU2 circuit pattern which consists of:
    1. Layers of single qubit SU(2) operations (RY, RZ rotations)
    2. Entanglement layers using CNOT gates
    3. Repeated structure for expressiveness
    
    The EfficientSU2 is ideal for molecular systems because:
    - Hardware-efficient design for NISQ devices
    - Good expressiveness with manageable parameter count
    - Proven performance on quantum chemistry problems
    """
    
    def __init__(self, num_qubits: int, num_electrons: Tuple[int, int]):
        """
        Initialize the EfficientSU2 ansatz builder.
        
        Args:
            num_qubits: Number of qubits (2 * active_orbitals)
            num_electrons: (alpha, beta) electrons in active space
        """
        self.num_qubits = num_qubits
        self.num_electrons = num_electrons
        self.num_orbitals = num_qubits // 2
        self.ansatz = None
        self.parameters = None
        
        print(f"EfficientSU2 Ansatz Initialized:")
        print(f"  Qubits: {self.num_qubits}")
        print(f"  Orbitals: {self.num_orbitals}")
        print(f"  Electrons: {self.num_electrons}")
    
    def create_hartree_fock_state(self) -> QuantumCircuit:
        """
        Create Hartree-Fock reference state for molecular system.
        
        Returns:
            QuantumCircuit with HF state preparation
        """
        hf_circuit = QuantumCircuit(self.num_qubits)
        
        n_alpha, n_beta = self.num_electrons
        
        # Fill alpha spin orbitals (first half of qubits)
        for i in range(n_alpha):
            hf_circuit.x(i)
        
        # Fill beta spin orbitals (second half of qubits)
        for i in range(n_beta):
            hf_circuit.x(self.num_orbitals + i)
        
        print(f"Hartree-Fock state created: {n_alpha}α + {n_beta}β electrons")
        return hf_circuit
    
    def create_efficient_su2_ansatz(self,
                                   su2_gates: Optional[List[str]] = None,
                                   entanglement: str = 'reverse_linear',
                                   reps: int = 2,
                                   include_initial_state: bool = True,
                                   skip_final_rotation_layer: bool = False,
                                   parameter_prefix: str = 'θ') -> QuantumCircuit:
        """
        Create EfficientSU2 ansatz optimized for molecular systems.
        
        Args:
            su2_gates: SU(2) gates for rotation layers ['ry', 'rz'] (default)
            entanglement: Entanglement pattern - options:
                         'reverse_linear' (default - same as full but fewer gates)
                         'linear' (conservative for large molecules)
                         'circular' (adds boundary coupling)
                         'full' (all-to-all coupling)
            reps: Number of repetition layers
            include_initial_state: Whether to prepend Hartree-Fock state
            skip_final_rotation_layer: Whether to skip final rotation layer
            parameter_prefix: Parameter naming prefix
            
        Returns:
            EfficientSU2 quantum circuit
        """
        print(f"Creating EfficientSU2 ansatz...")
        
        # Default molecular-friendly SU(2) gates
        if su2_gates is None:
            su2_gates = ['ry', 'rz']  # Standard choice for molecular systems
        
        # Optimize entanglement for system size
        if self.num_qubits > 12 and entanglement == 'reverse_linear':
            print(f"Large system detected ({self.num_qubits} qubits)")
            print(f"Recommendation: Consider 'linear' entanglement for reduced depth")
        
        # Create initial state if requested
        initial_state = None
        if include_initial_state:
            initial_state = self.create_hartree_fock_state()
        
        # Create EfficientSU2 circuit using function API
        base_ansatz = efficient_su2(
            num_qubits=self.num_qubits,
            su2_gates=su2_gates,
            entanglement=entanglement,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=True  # For visualization
        )
        
        # Combine with initial state if provided
        if initial_state is not None:
            combined_circuit = QuantumCircuit(self.num_qubits)
            combined_circuit.compose(initial_state, inplace=True)
            combined_circuit.barrier()
            combined_circuit.compose(base_ansatz, inplace=True)
            base_ansatz = combined_circuit
        
        self.ansatz = base_ansatz
        self.parameters = base_ansatz.parameters
        
        # Display circuit properties
        print(f"EfficientSU2 ansatz created:")
        print(f"  SU(2) gates: {su2_gates}")
        print(f"  Entanglement: {entanglement}")
        print(f"  Repetitions: {reps}")
        print(f"  Parameters: {base_ansatz.num_parameters}")
        print(f"  Circuit depth: {base_ansatz.decompose().depth()}")
        print(f"  Gate count: {len(base_ansatz.decompose())}")
        
        return base_ansatz
    
    def analyze_entanglement_options(self) -> dict:
        """
        Analyze different entanglement patterns for the molecular system.
        
        Returns:
            Dictionary with analysis of entanglement options
        """
        entanglement_options = {
            'reverse_linear': {
                'description': 'Same unitary as full with fewer gates (IBM recommended)',
                'gates_per_layer': self.num_qubits - 1,
                'recommended_for': 'Default choice - good balance',
                'depth_impact': 'Medium'
            },
            'linear': {
                'description': 'Sequential qubit coupling (0-1, 1-2, 2-3, ...)',
                'gates_per_layer': self.num_qubits - 1,
                'recommended_for': 'Large molecules, conservative choice',
                'depth_impact': 'Low'
            },
            'circular': {
                'description': 'Linear + additional coupling between first and last qubit',
                'gates_per_layer': self.num_qubits,
                'recommended_for': 'Medium molecules with periodic structure',
                'depth_impact': 'Medium'
            },
            'full': {
                'description': 'All-to-all qubit coupling',
                'gates_per_layer': self.num_qubits * (self.num_qubits - 1) // 2,
                'recommended_for': 'Small molecules only (< 6 qubits)',
                'depth_impact': 'Very High'
            }
        }
        
        print(f"\nEntanglement Analysis for {self.num_qubits}-qubit system:")
        print("-" * 60)
        
        for pattern, info in entanglement_options.items():
            print(f"{pattern.upper()}:")
            print(f"  Description: {info['description']}")
            print(f"  CNOT gates per layer: {info['gates_per_layer']}")
            print(f"  Recommended for: {info['recommended_for']}")
            print(f"  Depth impact: {info['depth_impact']}")
            print()
        
        # Recommendations for this system
        print("RECOMMENDATIONS FOR YOUR SYSTEM:")
        if self.num_qubits <= 8:
            print("  Primary: 'reverse_linear' or 'circular'")
            print("  Alternative: 'full' (if convergence issues)")
        elif self.num_qubits <= 16:
            print("  Primary: 'reverse_linear' (IBM recommended)")
            print("  Alternative: 'linear' (if depth is critical)")
        else:
            print("  Primary: 'linear' (conservative for large systems)")
            print("  Alternative: 'reverse_linear' (if more expressiveness needed)")
        
        return entanglement_options
    
    def recommend_parameters(self) -> dict:
        """
        Recommend optimal parameters for the molecular system.
        
        Returns:
            Dictionary with recommended parameters
        """
        # Base recommendations
        recommendations = {
            'su2_gates': ['ry', 'rz'],
            'reps': 2,
            'entanglement': 'reverse_linear',
            'skip_final_rotation_layer': False,
            'include_initial_state': True
        }
        
        # Adjust based on system size
        if self.num_qubits > 16:
            recommendations.update({
                'reps': 1,  # Reduce depth for large systems
                'entanglement': 'linear',
                'su2_gates': ['ry']  # Minimal gate set
            })
            print("Large system adjustments applied")
        elif self.num_qubits > 10:
            recommendations.update({
                'reps': 2,
                'entanglement': 'reverse_linear'
            })
        else:
            recommendations.update({
                'reps': 3,  # More layers for small systems
                'entanglement': 'reverse_linear'
            })
        
        # Estimate resources
        num_params = self._estimate_parameters(
            recommendations['reps'], 
            len(recommendations['su2_gates'])
        )
        depth = self._estimate_depth(
            recommendations['reps'],
            recommendations['entanglement']
        )
        
        recommendations.update({
            'estimated_parameters': num_params,
            'estimated_depth': depth
        })
        
        print(f"Parameter Recommendations:")
        print(f"  SU(2) gates: {recommendations['su2_gates']}")
        print(f"  Repetitions: {recommendations['reps']}")
        print(f"  Entanglement: {recommendations['entanglement']}")
        print(f"  Estimated parameters: {num_params}")
        print(f"  Estimated depth: {depth}")
        
        return recommendations
    
    def _estimate_parameters(self, reps: int, num_su2_gates: int) -> int:
        """Estimate number of parameters for given configuration."""
        # Each qubit gets num_su2_gates parameters per rotation layer
        # Number of rotation layers = reps + 1 (including final layer)
        params_per_layer = self.num_qubits * num_su2_gates
        rotation_layers = reps + 1
        return params_per_layer * rotation_layers
    
    def _estimate_depth(self, reps: int, entanglement: str) -> int:
        """Estimate circuit depth for given configuration."""
        # Rough depth estimation
        rotation_depth = 2  # Assuming 2 SU(2) gates per layer
        entanglement_depth = 2 if entanglement == 'full' else 1
        
        depth_per_rep = rotation_depth + entanglement_depth
        total_depth = rotation_depth + (reps * depth_per_rep)
        
        return total_depth
    
    def create_optimized_ansatz_for_amoxicillin(self) -> QuantumCircuit:
        """
        Create EfficientSU2 ansatz specifically optimized for amoxicillin.
        
        Based on system size (16 qubits) and molecular properties.
        
        Returns:
            Optimized EfficientSU2 circuit for amoxicillin
        """
        print("=" * 60)
        print("CREATING OPTIMIZED EFFICIENTSU2 FOR AMOXICILLIN")
        print("=" * 60)
        
        # Get recommendations
        recommendations = self.recommend_parameters()
        
        # Create optimized ansatz
        ansatz = self.create_efficient_su2_ansatz(
            su2_gates=recommendations['su2_gates'],
            entanglement=recommendations['entanglement'],
            reps=recommendations['reps'],
            include_initial_state=recommendations['include_initial_state'],
            skip_final_rotation_layer=False
        )
        
        print("=" * 60)
        print("AMOXICILLIN EFFICIENTSU2 ANSATZ READY")
        print("=" * 60)
        print(f"Circuit optimized for 16-qubit drug molecule")
        print(f"Ready for VQE optimization with {ansatz.num_parameters} parameters")
        
        return ansatz
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """
        Get reasonable parameter bounds for molecular VQE optimization.
        
        Returns:
            List of (min, max) bounds for each parameter
        """
        if self.ansatz is None:
            raise ValueError("Must create ansatz first")
        
        # Standard bounds for rotation gates in molecular systems
        # RY and RZ gates typically use [0, 2π] or [-π, π]
        bounds = []
        for param in self.parameters:
            if 'ry' in param.name.lower() or 'rz' in param.name.lower():
                bounds.append((0, 2 * np.pi))
            else:
                bounds.append((-np.pi, np.pi))  # Conservative default
        
        print(f"Parameter bounds set for {len(bounds)} parameters")
        return bounds
    
    def draw_ansatz(self, output: str = 'text', fold: int = 120) -> str:
        """
        Draw the current ansatz circuit.
        
        Args:
            output: 'text' or 'mpl' for matplotlib
            fold: Line width for text output
            
        Returns:
            Circuit drawing
        """
        if self.ansatz is None:
            return "No ansatz created yet"
        
        print("Circuit diagram:")
        if output == 'text':
            return self.ansatz.draw(output=output, fold=fold)
        else:
            return self.ansatz.draw(output=output)
    
    def create_circuit_visualization(self, style: str = 'iqp', save_path: Optional[str] = None):
        """
        Create a matplotlib visualization of the EfficientSU2 circuit.
        
        Args:
            style: Drawing style ('iqp', 'textbook', 'default')
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.ansatz is None:
            print("No ansatz created yet")
            return None
        
        try:
            import matplotlib.pyplot as plt
            from qiskit.visualization import circuit_drawer
            
            # Create the matplotlib figure
            fig = circuit_drawer(
                self.ansatz, 
                output='mpl',
                style=style,
                fold=20  # Adjust folding for readability
            )
            
            # Customize the figure
            if hasattr(fig, 'suptitle'):
                fig.suptitle(f'EfficientSU2 Ansatz for Molecular VQE\n'
                           f'{self.num_qubits} qubits, {self.ansatz.num_parameters} parameters',
                           fontsize=14, fontweight='bold')
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Circuit saved to {save_path}")
            
            return fig
            
        except ImportError:
            print("Matplotlib not available for circuit visualization")
            return None
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None


def create_amoxicillin_efficient_su2() -> EfficientSU2MolecularAnsatz:
    """
    Create EfficientSU2 ansatz specifically for amoxicillin molecule.
    
    Returns:
        Configured EfficientSU2 ansatz builder with optimized circuit
    """
    print("AMOXICILLIN EFFICIENTSU2 ANSATZ CREATION")
    print("=" * 50)
    
    # Amoxicillin system parameters from your Hamiltonian results
    num_qubits = 16
    num_electrons = (8, 8)  # Closed shell
    
    # Create ansatz builder
    builder = EfficientSU2MolecularAnsatz(num_qubits, num_electrons)
    
    # Analyze entanglement options
    builder.analyze_entanglement_options()
    
    # Create optimized ansatz
    ansatz = builder.create_optimized_ansatz_for_amoxicillin()
    
    # Display final circuit info
    print(f"\nFinal Circuit Properties:")
    print(f"  Ansatz type: EfficientSU2")
    print(f"  Parameters: {ansatz.num_parameters}")
    print(f"  Depth: {ansatz.decompose().depth()}")
    print(f"  Gate count: {len(ansatz.decompose())}")
    
    return builder


if __name__ == "__main__":
    # Create EfficientSU2 ansatz for amoxicillin
    builder = create_amoxicillin_efficient_su2()
    
    # Show parameter bounds
    bounds = builder.get_parameter_bounds()
    print(f"\nParameter optimization bounds: {len(bounds)} parameters")
    
    # Draw circuit (text version for console)
    print("\nCircuit Structure (Text):")
    print(builder.draw_ansatz(fold=100))
    
    # Create matplotlib visualization
    print("\nCreating matplotlib circuit visualization...")
    fig = builder.create_circuit_visualization(
        style='iqp',  # IBM Quantum style similar to your example
        save_path='amoxicillin_efficient_su2_circuit.png'
    )
    
    if fig is not None:
        print("Circuit visualization created successfully!")
        print("Use plt.show() to display or check the saved PNG file")
        try:
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.show()
        except:
            print("Display not available, but circuit saved to PNG file")