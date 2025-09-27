"""
VQE Molecular Simulator - Complete Implementation
Consolidated VQE pipeline for molecular simulation from CSV coordinates

Usage: python vqe_molecular_simulator.py <csv_file_path>

Input: CSV file with columns: Atom,x,y,z
Output: 
  - Energy convergence PNG plot
  - Energy convergence CSV file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
from scipy.optimize import minimize

# PySCF for quantum chemistry
try:
    from pyscf import gto, scf, mcscf, ao2mo
    PYSCF_AVAILABLE = True
except ImportError:
    print("Error: PySCF not available. Please install with: pip install pyscf")
    PYSCF_AVAILABLE = False

# Qiskit for quantum simulation
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import StatevectorSimulator
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    QISKIT_AVAILABLE = True
except ImportError:
    print("Error: Qiskit not available. Please install with: pip install qiskit qiskit-aer")
    QISKIT_AVAILABLE = False


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


class MolecularVQESimulator:
    """
    Complete VQE simulation pipeline from CSV coordinates to results.
    """
    
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6):
        """
        Initialize the molecular VQE simulator.
        
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
        self.start_time = None
        
        print("Molecular VQE Simulator Initialized")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Tolerance: {tolerance}")
        print(f"  Backend: {self.backend.name}")
    
    def parse_csv_geometry(self, csv_path: str) -> Tuple[List, Dict]:
        """
        Parse molecular geometry from CSV file.
        
        Args:
            csv_path: Path to CSV file with Atom,x,y,z format
            
        Returns:
            Tuple of (atom_list, properties)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Reading molecular geometry from: {csv_path}")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Validate required columns
        required_cols = ['atom', 'x', 'y', 'z']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Build atom list
        atom_list = []
        for _, row in df.iterrows():
            atom = str(row['atom']).strip()
            coords = (float(row['x']), float(row['y']), float(row['z']))
            atom_list.append([atom, coords])
        
        # Analyze molecule composition
        atom_types = {}
        for atom, _ in atom_list:
            atom_types[atom] = atom_types.get(atom, 0) + 1
        
        print(f"Parsed {len(atom_list)} atoms")
        print("Molecule composition:")
        for atom_type, count in sorted(atom_types.items()):
            print(f"  {atom_type}: {count}")
        
        properties = {
            "atom_count": len(atom_list),
            "composition": atom_types
        }
        
        return atom_list, properties
    
    def build_molecule(self, atom_list: List, charge: int = 0, spin: int = 0) -> gto.Mole:
        """
        Build PySCF molecule object.
        
        Args:
            atom_list: List of [atom, (x, y, z)] coordinates
            charge: Molecular charge
            spin: Number of unpaired electrons
            
        Returns:
            PySCF molecule object
        """
        if not PYSCF_AVAILABLE:
            raise ImportError("PySCF required for molecular calculations")
        
        print(f"Building molecule with {len(atom_list)} atoms using STO-3G basis")
        
        mol = gto.Mole()
        mol.build(
            verbose=0,
            atom=atom_list,
            basis="sto-3g",
            charge=charge,
            spin=spin
        )
        
        print(f"Molecule properties:")
        print(f"  Total electrons: {mol.nelectron}")
        print(f"  Total orbitals: {mol.nao_nr()}")
        print(f"  Nuclear repulsion: {mol.energy_nuc():.6f} Hartree")
        
        return mol
    
    def run_scf(self, mol: gto.Mole) -> scf.hf.SCF:
        """Run SCF calculation."""
        print("Running SCF calculation...")
        
        if mol.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
        
        mf.max_cycle = 50
        mf.conv_tol = 1e-6
        
        energy = mf.kernel()
        
        if not mf.converged:
            print("Warning: SCF did not converge!")
        
        print(f"SCF energy: {energy:.6f} Hartree")
        return mf
    
    def determine_active_space(self, mol: gto.Mole) -> Tuple[int, Tuple[int, int]]:
        """
        Automatically determine active space based on molecular size.
        
        Args:
            mol: PySCF molecule object
            
        Returns:
            Tuple of (ncas, nelecas)
        """
        total_orbitals = mol.nao_nr()
        total_electrons = mol.nelectron
        
        # Automatic active space selection based on system size
        if total_orbitals > 50:
            ncas = 6  # Very conservative for large molecules
        elif total_orbitals > 30:
            ncas = 8
        elif total_orbitals > 15:
            ncas = 10
        else:
            ncas = min(8, total_orbitals)
        
        # Ensure we don't exceed system limits
        ncas = min(ncas, total_orbitals)
        
        # Distribute electrons in active space
        if mol.spin == 0:
            active_electrons = min(ncas * 2, total_electrons)
            nelecas = (active_electrons // 2, active_electrons // 2)
        else:
            alpha_excess = mol.spin
            active_electrons = min(ncas * 2, total_electrons)
            alpha_electrons = (active_electrons + alpha_excess) // 2
            beta_electrons = active_electrons - alpha_electrons
            nelecas = (alpha_electrons, beta_electrons)
        
        print(f"Active space configuration:")
        print(f"  Active orbitals: {ncas}")
        print(f"  Active electrons: {nelecas}")
        print(f"  Qubits needed: {2 * ncas}")
        
        return ncas, nelecas
    
    def setup_active_space(self, mf: scf.hf.SCF, ncas: int, nelecas: Tuple[int, int]) -> mcscf.CASCI:
        """Set up active space calculation."""
        print("Setting up active space calculation...")
        
        cas = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
        
        # Select orbitals around HOMO-LUMO gap
        total_electrons = mf.mol.nelectron
        homo_idx = total_electrons // 2 - 1
        start_idx = max(0, homo_idx - ncas // 2 + 1)
        active_orbitals = list(range(start_idx, start_idx + ncas))
        
        print(f"Selected active orbitals: {active_orbitals}")
        
        mo = cas.sort_mo(active_orbitals, base=0)
        energy = cas.kernel(mo)
        
        print(f"CASCI energy: {cas.e_tot:.6f} Hartree")
        
        return cas
    
    def jordan_wigner_transformation(self, n_qubits: int) -> Tuple[List[SparsePauliOp], List[SparsePauliOp]]:
        """
        Generate creation and annihilation operators using Jordan-Wigner transformation.
        
        Args:
            n_qubits: Number of qubits (spin orbitals)
            
        Returns:
            Tuple of (creation_operators, annihilation_operators)
        """
        creation_ops = []
        
        for p in range(n_qubits):
            pauli_string = ["I"] * n_qubits
            
            # Add Z operators for anticommutation
            for k in range(p):
                pauli_string[k] = "Z"
            
            # Creation operator: c† = (X + iY)/2
            pauli_string[p] = "X"
            x_string = "".join(reversed(pauli_string))
            
            pauli_string[p] = "Y"
            y_string = "".join(reversed(pauli_string))
            
            c_p = SparsePauliOp.from_list([
                (x_string, 0.5),
                (y_string, 0.5j)
            ])
            creation_ops.append(c_p)
        
        # Annihilation operators are Hermitian conjugates
        annihilation_ops = [c_op.adjoint() for c_op in creation_ops]
        
        return creation_ops, annihilation_ops
    
    def cholesky_decomposition(self, V: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, int]:
        """
        Cholesky decomposition for two-electron integrals.
        
        Args:
            V: Two-electron integral tensor
            eps: Convergence threshold
            
        Returns:
            Tuple of (L, ng) where L are Cholesky vectors and ng is the number of vectors
        """
        no = V.shape[0]
        chmax = 20 * no
        ng = 0
        
        W = V.reshape(no**2, no**2)
        L = np.zeros((no**2, chmax))
        Dmax = np.diagonal(W).copy()
        
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
        
        while vmax > eps and ng < chmax:
            L[:, ng] = W[:, nu_max]
            
            if ng > 0:
                L[:, ng] -= np.dot(L[:, 0:ng], L.T[0:ng, nu_max])
            
            if vmax <= 0:
                break
                
            L[:, ng] /= np.sqrt(vmax)
            Dmax -= L[:, ng] ** 2
            
            ng += 1
            nu_max = np.argmax(Dmax)
            vmax = Dmax[nu_max]
        
        L = L[:, :ng].reshape((no, no, ng))
        return L, ng
    
    def build_molecular_hamiltonian(self, cas: mcscf.CASCI) -> SparsePauliOp:
        """
        Build molecular Hamiltonian using Jordan-Wigner transformation.
        
        Args:
            cas: CASCI object with active space
            
        Returns:
            SparsePauliOp representing the molecular Hamiltonian
        """
        print("Building molecular Hamiltonian...")
        
        # Get effective Hamiltonian in active space
        h1e, ecore = cas.get_h1eff()
        h2e = ao2mo.restore(1, cas.get_h2eff(), cas.ncas)
        
        ncas = cas.ncas
        n_qubits = 2 * ncas
        
        print(f"Active space: {ncas} orbitals, {n_qubits} qubits")
        
        # Get Jordan-Wigner operators
        C, D = self.jordan_wigner_transformation(n_qubits)
        
        # Build excitation operators
        Exc = []
        for p in range(ncas):
            # Number operators for orbital p
            n_p = C[p] @ D[p] + C[ncas + p] @ D[ncas + p]
            Excp = [n_p]
            
            # Excitation operators between orbitals p and r
            for r in range(p + 1, ncas):
                exc_pr = (C[p] @ D[r] + C[ncas + p] @ D[ncas + r] + 
                         C[r] @ D[p] + C[ncas + r] @ D[ncas + p])
                Excp.append(exc_pr)
            
            Exc.append(Excp)
        
        # Cholesky decomposition for two-electron terms
        L_chol, ng = self.cholesky_decomposition(h2e, eps=1e-6)
        
        # Modified one-electron integrals
        t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
        
        # Initialize Hamiltonian with core energy
        H = ecore * SparsePauliOp.from_list([("I" * n_qubits, 1.0)])
        
        # Add one-electron terms
        for p in range(ncas):
            for r in range(p, ncas):
                coeff = t1e[p, r]
                if abs(coeff) > 1e-12:
                    H += coeff * Exc[p][r - p]
        
        # Add two-electron terms using Cholesky decomposition
        for g in range(ng):
            L_g = 0 * SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
            
            for p in range(ncas):
                for r in range(p, ncas):
                    coeff = L_chol[p, r, g]
                    if abs(coeff) > 1e-12:
                        L_g += coeff * Exc[p][r - p]
            
            H += 0.5 * (L_g @ L_g)
        
        H = H.chop(1e-12).simplify()
        
        print(f"Hamiltonian: {len(H)} terms, {H.num_qubits} qubits")
        
        return H
    
    def create_hartree_fock_state(self, num_qubits: int, nelecas: Tuple[int, int]) -> QuantumCircuit:
        """Create Hartree-Fock reference state."""
        hf_circuit = QuantumCircuit(num_qubits)
        
        n_alpha, n_beta = nelecas
        num_orbitals = num_qubits // 2
        
        # Fill alpha spin orbitals
        for i in range(n_alpha):
            hf_circuit.x(i)
        
        # Fill beta spin orbitals
        for i in range(n_beta):
            hf_circuit.x(num_orbitals + i)
        
        return hf_circuit
    
    def create_efficient_su2_ansatz(self, num_qubits: int, nelecas: Tuple[int, int], reps: int = 2) -> QuantumCircuit:
        """
        Create EfficientSU2 ansatz with Hartree-Fock initialization.
        
        Args:
            num_qubits: Number of qubits
            nelecas: (alpha, beta) electrons
            reps: Number of repetition layers
            
        Returns:
            Parameterized quantum circuit
        """
        print(f"Creating EfficientSU2 ansatz with {reps} repetitions...")
        
        # Create initial Hartree-Fock state
        hf_state = self.create_hartree_fock_state(num_qubits, nelecas)
        
        # Create parameter vector
        num_params = (reps + 1) * 2 * num_qubits  # RY and RZ gates
        params = ParameterVector('θ', num_params)
        
        # Build ansatz circuit
        ansatz = QuantumCircuit(num_qubits)
        
        # Add Hartree-Fock initialization
        ansatz.compose(hf_state, inplace=True)
        ansatz.barrier()
        
        param_idx = 0
        
        # Add rotation layers
        for rep in range(reps + 1):
            # RY rotation layer
            for qubit in range(num_qubits):
                ansatz.ry(params[param_idx], qubit)
                param_idx += 1
            
            # RZ rotation layer
            for qubit in range(num_qubits):
                ansatz.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Entanglement layer (except for last repetition)
            if rep < reps:
                for qubit in range(num_qubits - 1):
                    ansatz.cx(qubit, qubit + 1)
                ansatz.barrier()
        
        print(f"Ansatz created: {ansatz.num_parameters} parameters, depth {ansatz.decompose().depth()}")
        
        return ansatz
    
    def create_cost_function(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Callable:
        """
        Create the cost function for VQE optimization.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian
            
        Returns:
            Cost function for optimization
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum simulation")
        
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
                
                if self.evaluation_count % 10 == 0:
                    print(f"  Evaluation {self.evaluation_count}: Energy = {energy:.8f} Hartree")
                
                return energy
                
            except Exception as e:
                print(f"Warning: Evaluation {self.evaluation_count} failed: {e}")
                return 1e10
        
        return cost_function
    
    def optimize_vqe(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Tuple[float, np.ndarray]:
        """
        Run VQE optimization using COBYLA.
        
        Args:
            ansatz: Parameterized quantum circuit
            hamiltonian: Molecular Hamiltonian
            
        Returns:
            Tuple of (optimal_energy, optimal_parameters)
        """
        print("=" * 60)
        print("VQE OPTIMIZATION")
        print("=" * 60)
        
        # Reset tracking
        self.energy_history = []
        self.evaluation_count = 0
        
        # Create cost function
        cost_function = self.create_cost_function(ansatz, hamiltonian)
        
        # Initialize parameters
        num_parameters = ansatz.num_parameters
        initial_params = np.random.normal(0, 0.1, num_parameters)
        
        # Set parameter bounds
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
        
        # Run optimization
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
        
        print(f"Optimization completed: Energy = {result.fun:.8f} Hartree")
        print(f"Success: {result.success}")
        print(f"Function evaluations: {result.nfev}")
        
        return result.fun, result.x
    
    def save_energy_convergence_csv(self, csv_path: str, base_filename: str):
        """
        Save energy convergence data to CSV file.
        
        Args:
            csv_path: Original CSV file path
            base_filename: Base filename for output
        """
        # Create output filename
        output_filename = f"{base_filename}_energy.csv"
        
        # Create DataFrame
        df = pd.DataFrame({
            'Function Evaluation': range(1, len(self.energy_history) + 1),
            'Energy (Hartree)': self.energy_history
        })
        
        # Save to CSV
        df.to_csv(output_filename, index=False)
        print(f"Energy convergence saved to: {output_filename}")
    
    def create_visualization(self, result: VQEResult, base_filename: str):
        """
        Create comprehensive visualization of VQE results.
        
        Args:
            result: VQE simulation result
            base_filename: Base filename for output
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'VQE Simulation Results\n{result.molecular_formula}', 
                    fontsize=16, fontweight='bold')
        
        # Energy convergence
        axes[0, 0].plot(range(1, len(result.energy_history) + 1), 
                       result.energy_history, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].axhline(y=result.ground_state_energy, color='r', linestyle='--', 
                          label=f'Final: {result.ground_state_energy:.6f}')
        axes[0, 0].set_xlabel('Function Evaluation')
        axes[0, 0].set_ylabel('Energy (Hartree)')
        axes[0, 0].set_title('Energy Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Energy improvement
        if len(result.energy_history) > 1:
            energy_array = np.array(result.energy_history)
            running_best = np.minimum.accumulate(energy_array)
            axes[0, 1].plot(range(1, len(running_best) + 1), running_best, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Function Evaluation')
            axes[0, 1].set_ylabel('Best Energy So Far (Hartree)')
            axes[0, 1].set_title('Best Energy Progress')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Parameter values (first 8 parameters)
        num_params_to_plot = min(8, len(result.optimal_parameters))
        x_pos = np.arange(num_params_to_plot)
        axes[1, 0].bar(x_pos, result.optimal_parameters[:num_params_to_plot], 
                      color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Parameter Index')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title(f'Optimal Parameters (first {num_params_to_plot})')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].grid(True, alpha=0.3)
        
        # System information
        axes[1, 1].axis('off')
        info_text = f"""
System Information:
• Qubits: {result.num_qubits}
• Parameters: {result.num_parameters}
• Evaluations: {result.evaluation_count}
• Execution Time: {result.execution_time:.2f}s
• Success: {result.success}

Final Energy:
{result.ground_state_energy:.8f} Hartree

Molecular Formula:
{result.molecular_formula}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                        verticalalignment='center')
        
        plt.tight_layout()
        
        # Save plot
        output_filename = f"{base_filename}_vqe_results.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_filename}")
        
        return fig
    
    def run_simulation(self, csv_path: str) -> VQEResult:
        """
        Run complete VQE simulation from CSV file.
        
        Args:
            csv_path: Path to molecular coordinates CSV file
            
        Returns:
            VQE simulation result
        """
        print("=" * 70)
        print("MOLECULAR VQE SIMULATION")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Parse molecular geometry
        atom_list, properties = self.parse_csv_geometry(csv_path)
        
        # Build molecule
        mol = self.build_molecule(atom_list)
        
        # Run SCF
        mf = self.run_scf(mol)
        
        # Determine active space
        ncas, nelecas = self.determine_active_space(mol)
        
        # Setup active space calculation
        cas = self.setup_active_space(mf, ncas, nelecas)
        
        # Build Hamiltonian
        hamiltonian = self.build_molecular_hamiltonian(cas)
        
        # Create ansatz
        num_qubits = hamiltonian.num_qubits
        ansatz = self.create_efficient_su2_ansatz(num_qubits, nelecas)
        
        # Run VQE optimization
        optimal_energy, optimal_params = self.optimize_vqe(ansatz, hamiltonian)
        
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
            energy_history=self.energy_history.copy(),
            evaluation_count=self.evaluation_count,
            execution_time=execution_time,
            molecular_formula=molecular_formula,
            num_qubits=num_qubits,
            num_parameters=len(optimal_params),
            success=optimal_energy < 1e9
        )
        
        # Generate output files
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        
        # Save energy convergence CSV
        self.save_energy_convergence_csv(csv_path, base_filename)
        
        # Create visualization
        self.create_visualization(result, base_filename)
        
        # Display summary
        print("\n" + "=" * 70)
        print("VQE SIMULATION COMPLETED")
        print("=" * 70)
        print(f"Molecular formula: {molecular_formula}")
        print(f"Ground state energy: {optimal_energy:.8f} Hartree")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Function evaluations: {self.evaluation_count}")
        print(f"Success: {result.success}")
        print("=" * 70)
        
        return result


def main():
    """Main function to run VQE simulation from command line."""
    if len(sys.argv) != 2:
        print("Usage: python vqe_molecular_simulator.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found")
        sys.exit(1)
    
    # Check dependencies
    if not PYSCF_AVAILABLE:
        print("Error: PySCF not installed. Please install with: pip install pyscf")
        sys.exit(1)
    
    if not QISKIT_AVAILABLE:
        print("Error: Qiskit not installed. Please install with: pip install qiskit qiskit-aer")
        sys.exit(1)
    
    try:
        # Create simulator with specified parameters
        simulator = MolecularVQESimulator(
            max_iterations=10,
            tolerance=1e-6
        )
        
        # Run complete simulation
        result = simulator.run_simulation(csv_path)
        
        print("\nSimulation completed successfully!")
        print(f"Output files created:")
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"  - {base_filename}_energy.csv (energy convergence data)")
        print(f"  - {base_filename}_vqe_results.png (visualization)")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()