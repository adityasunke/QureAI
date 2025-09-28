"""
EfficientSU2 Ansatz Generator for Any Molecule
Creates parameterized quantum circuits for VQE optimization
"""

import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def create_hartree_fock_state(num_qubits: int, num_electrons: Tuple[int, int]) -> QuantumCircuit:
    """
    Create Hartree-Fock reference state.
    
    Args:
        num_qubits: Number of qubits
        num_electrons: (alpha, beta) electrons in active space
        
    Returns:
        QuantumCircuit with HF state preparation
    """
    hf_circuit = QuantumCircuit(num_qubits)
    
    n_alpha, n_beta = num_electrons
    num_orbitals = num_qubits // 2
    
    # Fill alpha spin orbitals (first half of qubits)
    for i in range(n_alpha):
        hf_circuit.x(i)
    
    # Fill beta spin orbitals (second half of qubits)
    for i in range(n_beta):
        hf_circuit.x(num_orbitals + i)
    
    return hf_circuit


def create_ansatz(num_qubits: int, num_electrons: Tuple[int, int], reps: int = 2) -> QuantumCircuit:
    """
    Create EfficientSU2 ansatz with Hartree-Fock initialization.
    
    Args:
        num_qubits: Number of qubits
        num_electrons: (alpha, beta) electrons in active space
        reps: Number of repetition layers
        
    Returns:
        Parameterized quantum circuit
    """
    print(f"Creating EfficientSU2 ansatz:")
    print(f"  Qubits: {num_qubits}")
    print(f"  Electrons: {num_electrons}")
    print(f"  Repetitions: {reps}")
    
    # Create initial Hartree-Fock state
    hf_state = create_hartree_fock_state(num_qubits, num_electrons)
    
    # Create parameter vector for rotation gates
    # Each layer has RY and RZ gates for each qubit
    num_params = (reps + 1) * 2 * num_qubits
    params = ParameterVector('theta', num_params)
    
    # Build main ansatz circuit
    ansatz = QuantumCircuit(num_qubits)
    
    # Add Hartree-Fock initialization
    ansatz.compose(hf_state, inplace=True)
    ansatz.barrier()
    
    param_idx = 0
    
    # Add EfficientSU2 layers
    for rep in range(reps + 1):
        # Rotation layer: RY gates
        for qubit in range(num_qubits):
            ansatz.ry(params[param_idx], qubit)
            param_idx += 1
        
        # Rotation layer: RZ gates
        for qubit in range(num_qubits):
            ansatz.rz(params[param_idx], qubit)
            param_idx += 1
        
        # Entanglement layer (linear connectivity, except for last repetition)
        if rep < reps:
            for qubit in range(num_qubits - 1):
                ansatz.cx(qubit, qubit + 1)
            ansatz.barrier()
    
    print(f"Ansatz created:")
    print(f"  Parameters: {ansatz.num_parameters}")
    print(f"  Circuit depth: {ansatz.decompose().depth()}")
    
    return ansatz


if __name__ == "__main__":
    # Test ansatz creation
    test_qubits = 8
    test_electrons = (4, 4)
    test_reps = 2
    
    print("Testing ansatz creation...")
    ansatz = create_ansatz(test_qubits, test_electrons, test_reps)
    
    print(f"\nTest completed:")
    print(f"Circuit has {ansatz.num_parameters} parameters")
    print(f"Circuit depth: {ansatz.decompose().depth()}")
