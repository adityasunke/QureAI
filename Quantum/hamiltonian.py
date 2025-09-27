"""
Molecular Hamiltonian Generator for Quantum Chemistry - CSV Implementation
Specifically designed for processing molecular geometries from CSV files
Compatible with latest Qiskit and PySCF versions
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
import os

# PySCF imports
from pyscf import gto, scf, mcscf, ao2mo

# Qiskit imports - updated for latest version
from qiskit.quantum_info import SparsePauliOp


def cholesky_decomposition(V: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, int]:
    """
    Cholesky decomposition for two-electron integrals with improved stability.
    
    Decomposes the 4D two-electron integral tensor V into a low-rank representation
    using Cholesky vectors for efficient quantum Hamiltonian construction.
    
    Args:
        V: Two-electron integral tensor (no, no, no, no)
        eps: Convergence threshold for decomposition
        
    Returns:
        Tuple of (L, ng) where L are Cholesky vectors and ng is the number of vectors
    """
    no = V.shape[0]
    chmax = 20 * no  # Maximum number of Cholesky vectors
    ng = 0           # Current number of vectors
    
    # Reshape to matrix form for decomposition
    W = V.reshape(no**2, no**2)
    L = np.zeros((no**2, chmax))
    Dmax = np.diagonal(W).copy()
    
    # Track maximum diagonal element
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    
    print(f"Starting Cholesky decomposition with threshold {eps}")
    
    # Iterative Cholesky with safety checks
    while vmax > eps and ng < chmax:
        # Extract column corresponding to maximum diagonal
        L[:, ng] = W[:, nu_max]
        
        # Subtract contributions from previous vectors
        if ng > 0:
            L[:, ng] -= np.dot(L[:, 0:ng], L.T[0:ng, nu_max])
        
        # Check for numerical stability
        if vmax <= 0:
            print(f"Warning: Non-positive diagonal element {vmax} at iteration {ng}")
            break
            
        # Normalize and update
        L[:, ng] /= np.sqrt(vmax)
        Dmax -= L[:, ng] ** 2
        
        ng += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    
    # Reshape back to tensor form
    L = L[:, :ng].reshape((no, no, ng))
    
    # Verify decomposition accuracy
    reconstruction = np.einsum('prg,qsg->prqs', L, L)
    max_error = np.abs(reconstruction - V).max()
    
    print(f"Cholesky decomposition completed:")
    print(f"  Vectors used: {ng}/{chmax}")
    print(f"  Max reconstruction error: {max_error:.2e}")
    
    if max_error > 1e-10:
        warnings.warn(f"Large reconstruction error: {max_error:.2e}")
    
    return L, ng


def create_identity_operator(n_qubits: int) -> SparsePauliOp:
    """Create identity operator for n qubits using latest Qiskit syntax."""
    return SparsePauliOp.from_list([("I" * n_qubits, 1.0)])


def jordan_wigner_transformation(n_qubits: int) -> Tuple[List[SparsePauliOp], List[SparsePauliOp]]:
    """
    Generate creation and annihilation operators using Jordan-Wigner transformation.
    Updated for latest Qiskit SparsePauliOp interface.
    
    The Jordan-Wigner transformation maps fermionic operators to qubit operators:
    c†_p = (X_p + iY_p)/2 × ∏_{k<p} Z_k
    c_p = (X_p - iY_p)/2 × ∏_{k<p} Z_k
    
    Args:
        n_qubits: Number of qubits (spin orbitals)
        
    Returns:
        Tuple of (creation_operators, annihilation_operators)
    """
    creation_ops = []
    
    for p in range(n_qubits):
        # Build Pauli string with Z operators for anticommutation
        pauli_string = ["I"] * n_qubits
        
        # Add Z operators for all qubits before p
        for k in range(p):
            pauli_string[k] = "Z"
        
        # Set the p-th qubit operator
        pauli_string[p] = "X"
        x_string = "".join(reversed(pauli_string))  # Qiskit uses reverse ordering
        
        pauli_string[p] = "Y"
        y_string = "".join(reversed(pauli_string))
        
        # Creation operator: c†_p = (X + iY)/2
        c_p = SparsePauliOp.from_list([
            (x_string, 0.5),
            (y_string, 0.5j)
        ])
        creation_ops.append(c_p)
    
    # Annihilation operators are Hermitian conjugates
    annihilation_ops = [c_op.adjoint() for c_op in creation_ops]
    
    return creation_ops, annihilation_ops


def build_molecular_hamiltonian(ecore: float, h1e: np.ndarray, h2e: np.ndarray) -> SparsePauliOp:
    """
    Build molecular Hamiltonian using Jordan-Wigner transformation.
    Updated with improved error handling and latest Qiskit features.
    
    Constructs the electronic Hamiltonian:
    H = E_core + Σ h_pq c†_p c_q + 1/2 Σ h_pqrs c†_p c†_q c_s c_r
    
    Args:
        ecore: Core energy (nuclear repulsion + frozen core)
        h1e: One-electron integrals (kinetic + nuclear attraction)
        h2e: Two-electron repulsion integrals
        
    Returns:
        SparsePauliOp representing the molecular Hamiltonian
    """
    print("Building molecular Hamiltonian...")
    
    ncas = h1e.shape[0]  # Number of active space orbitals
    n_qubits = 2 * ncas  # Each spatial orbital has 2 spin orbitals
    
    print(f"Active space: {ncas} orbitals, {n_qubits} qubits")
    
    # Get Jordan-Wigner operators
    C, D = jordan_wigner_transformation(n_qubits)
    
    # Build excitation operators for one and two-electron terms
    # Exc[p][0] = number operator n_p = c†_pα c_pα + c†_pβ c_pβ
    # Exc[p][r-p] = excitation operators between orbitals p and r
    Exc = []
    for p in range(ncas):
        # Number operators for orbital p (alpha and beta spin)
        n_p = C[p] @ D[p] + C[ncas + p] @ D[ncas + p]
        Excp = [n_p]
        
        # Excitation operators between orbitals p and r
        for r in range(p + 1, ncas):
            # c†_p c_r + c†_{p+ncas} c_{r+ncas} + h.c.
            exc_pr = (C[p] @ D[r] + C[ncas + p] @ D[ncas + r] + 
                     C[r] @ D[p] + C[ncas + r] @ D[ncas + p])
            Excp.append(exc_pr)
        
        Exc.append(Excp)
    
    # Cholesky decomposition for efficient two-electron terms
    print("Performing Cholesky decomposition...")
    L_chol, ng = cholesky_decomposition(h2e, eps=1e-6)
    
    # Modified one-electron integrals (subtract mean-field contribution)
    t1e = h1e - 0.5 * np.einsum("pxxr->pr", h2e)
    
    # Initialize Hamiltonian with core energy
    H = ecore * create_identity_operator(n_qubits)
    print(f"Core energy term: {ecore:.6f}")
    
    # Add one-electron terms
    print("Adding one-electron terms...")
    one_electron_terms = 0
    for p in range(ncas):
        for r in range(p, ncas):
            coeff = t1e[p, r]
            if abs(coeff) > 1e-12:  # Skip negligible terms
                H += coeff * Exc[p][r - p]
                one_electron_terms += 1
    
    print(f"Added {one_electron_terms} one-electron terms")
    
    # Add two-electron terms using Cholesky decomposition
    print("Adding two-electron terms...")
    for g in range(ng):
        # Build operator for Cholesky vector g
        L_g = 0 * create_identity_operator(n_qubits)
        
        for p in range(ncas):
            for r in range(p, ncas):
                coeff = L_chol[p, r, g]
                if abs(coeff) > 1e-12:
                    L_g += coeff * Exc[p][r - p]
        
        # Add squared term: 1/2 * L_g^2
        H += 0.5 * (L_g @ L_g)
    
    print(f"Added {ng} Cholesky vectors for two-electron terms")
    
    # Simplify the Hamiltonian
    H = H.chop(1e-12).simplify()
    
    print(f"Final Hamiltonian: {len(H)} terms, {H.num_qubits} qubits")
    
    return H


def parse_csv_geometry(csv_path: str) -> Tuple[List, Dict]:
    """
    Parse molecular geometry from CSV file.
    Optimized for the specific format: Atom, x, y, z
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (atom_list, properties)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Reading molecular geometry from: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"CSV shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean column names (remove spaces, convert to lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Extract atomic coordinates
    atom_list = []
    
    # Map column names to expected format
    atom_col = None
    coord_cols = None
    
    # Find atom column
    for col in df.columns:
        if col in ['atom', 'element', 'symbol', 'atomic_symbol']:
            atom_col = col
            break
    
    # Find coordinate columns
    if all(col in df.columns for col in ['x', 'y', 'z']):
        coord_cols = ['x', 'y', 'z']
    
    if atom_col is None:
        raise ValueError(f"Could not find atom column. Available: {list(df.columns)}")
    
    if coord_cols is None:
        raise ValueError(f"Could not find coordinate columns (x, y, z). Available: {list(df.columns)}")
    
    print(f"Using atom column: {atom_col}")
    print(f"Using coordinate columns: {coord_cols}")
    
    # Build atom list
    for _, row in df.iterrows():
        atom = str(row[atom_col]).strip()
        coords = (float(row['x']), float(row['y']), float(row['z']))
        atom_list.append([atom, coords])
    
    print(f"Parsed {len(atom_list)} atoms")
    
    # Show atom types and counts
    atom_types = {}
    for atom, _ in atom_list:
        atom_types[atom] = atom_types.get(atom, 0) + 1
    
    print("Molecule composition:")
    for atom_type, count in sorted(atom_types.items()):
        print(f"  {atom_type}: {count}")
    
    # Set default properties
    properties = {
        "charge": 0,
        "spin": 0,
        "basis": "sto-3g",
        "atom_count": len(atom_list),
        "composition": atom_types
    }
    
    return atom_list, properties


class QuantumChemistryHamiltonian:
    """
    Molecular Hamiltonian generator for quantum chemistry from CSV data.
    Optimized for processing large molecules like amoxicillin.
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the Hamiltonian builder.
        
        Args:
            csv_path: Path to CSV file containing molecular geometry
        """
        self.csv_path = csv_path
        self.mol = None
        self.mf = None
        self.cas = None
        self.hamiltonian = None
        self.csv_properties = {}
        
    def load_molecule_from_csv(self, csv_path: Optional[str] = None) -> Tuple[List, Dict]:
        """
        Load molecular geometry from CSV file.
        
        Args:
            csv_path: Path to CSV file (uses self.csv_path if None)
            
        Returns:
            Tuple of (atom_list, properties)
        """
        if csv_path is None:
            csv_path = self.csv_path
        
        if csv_path is None:
            raise ValueError("No CSV path provided")
        
        atom_list, properties = parse_csv_geometry(csv_path)
        self.csv_properties = properties
        
        return atom_list, properties
    
    def build_molecule_from_csv(self, csv_path: Optional[str] = None,
                               basis: str = "sto-3g",
                               charge: int = 0,
                               spin: int = 0) -> gto.Mole:
        """
        Build PySCF molecule from CSV data.
        
        Args:
            csv_path: Path to CSV file
            basis: Basis set for calculation
            charge: Molecular charge
            spin: Number of unpaired electrons
            
        Returns:
            PySCF molecule object
        """
        # Load from CSV
        atom_list, properties = self.load_molecule_from_csv(csv_path)
        
        # Build molecule
        return self.build_molecule(
            atom_list=atom_list,
            basis=basis,
            charge=charge,
            spin=spin
        )
    
    def build_molecule(self, atom_list: List, basis: str = "sto-3g", 
                      charge: int = 0, spin: int = 0, symmetry: bool = False) -> gto.Mole:
        """
        Build PySCF molecule object.
        
        Args:
            atom_list: List of [atom, (x, y, z)] coordinates
            basis: Basis set for calculation
            charge: Molecular charge
            spin: Number of unpaired electrons
            symmetry: Use molecular symmetry
            
        Returns:
            PySCF molecule object
        """
        print(f"Building molecule with {len(atom_list)} atoms using {basis} basis")
        
        self.mol = gto.Mole()
        self.mol.build(
            verbose=1,
            atom=atom_list,
            basis=basis,
            charge=charge,
            spin=spin,
            symmetry=symmetry
        )
        
        print(f"Molecule properties:")
        print(f"  Total electrons: {self.mol.nelectron}")
        print(f"  Total orbitals: {self.mol.nao_nr()}")
        print(f"  Nuclear repulsion: {self.mol.energy_nuc():.6f} Hartree")
        
        return self.mol
    
    def run_scf(self, max_cycle: int = 50, conv_tol: float = 1e-6) -> float:
        """Run SCF calculation."""
        if self.mol is None:
            raise ValueError("Must build molecule first")
        
        print("Running SCF calculation...")
        
        # Choose appropriate SCF method
        if self.mol.spin == 0:
            self.mf = scf.RHF(self.mol)
        else:
            self.mf = scf.UHF(self.mol)
        
        # Set convergence parameters
        self.mf.max_cycle = max_cycle
        self.mf.conv_tol = conv_tol
        
        # Run calculation
        energy = self.mf.kernel()
        
        if not self.mf.converged:
            warnings.warn("SCF did not converge!")
        
        print(f"SCF energy: {energy:.6f} Hartree")
        return energy
    
    def setup_active_space(self, ncas: Optional[int] = None, 
                          nelecas: Optional[Tuple[int, int]] = None, 
                          orbital_selection: str = "homo_lumo") -> mcscf.CASCI:
        """
        Set up active space calculation with automatic parameter detection.
        For large molecules like amoxicillin, uses conservative active space.
        
        Args:
            ncas: Number of active space orbitals (auto-detect if None)
            nelecas: (alpha, beta) electrons in active space (auto-detect if None)
            orbital_selection: Strategy for orbital selection
            
        Returns:
            CASCI object
        """
        if self.mf is None:
            raise ValueError("Must run SCF first")
        
        total_orbitals = self.mol.nao_nr()
        total_electrons = self.mol.nelectron
        
        # Auto-detect active space parameters for large molecules
        if ncas is None:
            if total_orbitals > 50:  # Large molecule like amoxicillin
                ncas = min(8, total_orbitals)  # Conservative choice
                print(f"Large molecule detected ({total_orbitals} orbitals). Using conservative active space.")
            elif total_orbitals > 20:
                ncas = min(10, total_orbitals)
            elif total_orbitals > 10:
                ncas = min(8, total_orbitals)
            else:
                ncas = min(6, total_orbitals)
        
        if nelecas is None:
            # Distribute electrons in active space
            if self.mol.spin == 0:
                active_electrons = min(ncas * 2, total_electrons)
                nelecas = (active_electrons // 2, active_electrons // 2)
            else:
                alpha_excess = self.mol.spin
                active_electrons = min(ncas * 2, total_electrons)
                alpha_electrons = (active_electrons + alpha_excess) // 2
                beta_electrons = active_electrons - alpha_electrons
                nelecas = (alpha_electrons, beta_electrons)
        
        print(f"Active space configuration:")
        print(f"  Total orbitals: {total_orbitals}")
        print(f"  Active orbitals: {ncas}")
        print(f"  Active electrons: {nelecas}")
        print(f"  Qubits needed: {2 * ncas}")
        
        self.cas = mcscf.CASCI(self.mf, ncas=ncas, nelecas=nelecas)
        
        # Select active space orbitals
        if orbital_selection == "homo_lumo":
            # Select orbitals around HOMO-LUMO gap
            homo_idx = total_electrons // 2 - 1
            start_idx = max(0, homo_idx - ncas // 2 + 1)
            active_orbitals = list(range(start_idx, start_idx + ncas))
        else:
            # Default to first ncas orbitals
            active_orbitals = list(range(ncas))
        
        print(f"Selected active orbitals: {active_orbitals}")
        
        # Sort and run CASCI
        mo = self.cas.sort_mo(active_orbitals, base=0)
        energy = self.cas.kernel(mo)
        
        print(f"CASCI energy: {self.cas.e_tot:.6f} Hartree")
        
        return self.cas
    
    def generate_hamiltonian(self) -> SparsePauliOp:
        """Generate the molecular Hamiltonian for quantum computation."""
        if self.cas is None:
            raise ValueError("Must set up active space first")
        
        print("Extracting Hamiltonian components...")
        
        # Get effective Hamiltonian in active space
        h1e, ecore = self.cas.get_h1eff()
        h2e = ao2mo.restore(1, self.cas.get_h2eff(), self.cas.ncas)
        
        print(f"Core energy: {ecore:.6f} Hartree")
        print(f"Active space size: {self.cas.ncas} orbitals")
        
        # Build quantum Hamiltonian
        self.hamiltonian = build_molecular_hamiltonian(ecore, h1e, h2e)
        
        return self.hamiltonian
    
    def process_csv_molecule(self, csv_path: Optional[str] = None, 
                           basis: str = "sto-3g",
                           charge: int = 0,
                           spin: int = 0,
                           active_space_params: Optional[Dict] = None) -> SparsePauliOp:
        """
        Complete workflow: CSV → Hamiltonian in one function.
        
        Args:
            csv_path: Path to CSV file
            basis: Basis set for calculation
            charge: Molecular charge
            spin: Number of unpaired electrons
            active_space_params: Parameters for active space setup
            
        Returns:
            Molecular Hamiltonian as SparsePauliOp
        """
        if csv_path is None:
            csv_path = self.csv_path
            
        print(f"=== Processing molecule from CSV: {csv_path} ===")
        
        # Build molecule from CSV
        self.build_molecule_from_csv(csv_path, basis=basis, charge=charge, spin=spin)
        
        # Run SCF
        self.run_scf()
        
        # Set up active space
        if active_space_params is None:
            active_space_params = {}
        
        self.setup_active_space(**active_space_params)
        
        # Generate Hamiltonian
        hamiltonian = self.generate_hamiltonian()
        
        return hamiltonian
    
    def get_properties(self) -> Dict:
        """Get comprehensive molecular and calculation properties."""
        properties = {}
        
        # Add CSV properties
        properties.update(self.csv_properties)
        
        if self.mol is not None:
            properties.update({
                "total_electrons": self.mol.nelectron,
                "total_orbitals": self.mol.nao_nr(),
                "nuclear_repulsion": self.mol.energy_nuc(),
            })
        
        if self.mf is not None:
            properties.update({
                "scf_energy": self.mf.energy_tot(),
                "scf_converged": self.mf.converged,
            })
        
        if self.cas is not None:
            properties.update({
                "active_space_orbitals": self.cas.ncas,
                "active_space_electrons": self.cas.nelecas,
                "casci_energy": self.cas.e_tot,
            })
        
        if self.hamiltonian is not None:
            properties.update({
                "hamiltonian_qubits": self.hamiltonian.num_qubits,
                "hamiltonian_terms": len(self.hamiltonian),
            })
        
        return properties


def process_amoxicillin(csv_path: str = "amoxicillin_Nm.csv"):
    """
    Process amoxicillin molecule from CSV and generate Hamiltonian.
    
    Args:
        csv_path: Path to amoxicillin CSV file
        
    Returns:
        Tuple of (hamiltonian, properties)
    """
    print("=== Amoxicillin Molecule Hamiltonian Generation ===")
    
    # Initialize builder
    builder = QuantumChemistryHamiltonian(csv_path)
    
    # Process with appropriate parameters for a large organic molecule
    print("Note: Amoxicillin is a large molecule. Using conservative active space.")
    
    # Generate Hamiltonian with optimized parameters
    hamiltonian = builder.process_csv_molecule(
        csv_path,
        basis="sto-3g",  # Minimal basis for large molecule
        charge=0,        # Neutral molecule
        spin=0           # Likely closed shell
    )
    
    # Get properties
    properties = builder.get_properties()
    
    # Display results
    print(f"\n{'='*60}")
    print("AMOXICILLIN MOLECULE ANALYSIS")
    print(f"{'='*60}")
    
    print("\nMolecular Properties:")
    print(f"  Formula: {properties.get('composition', 'Unknown')}")
    print(f"  Total atoms: {properties.get('atom_count', 'Unknown')}")
    print(f"  Total electrons: {properties.get('total_electrons', 'Unknown')}")
    print(f"  Total orbitals: {properties.get('total_orbitals', 'Unknown')}")
    print(f"  SCF energy: {properties.get('scf_energy', 'Unknown'):.6f} Hartree")
    print(f"  CASCI energy: {properties.get('casci_energy', 'Unknown'):.6f} Hartree")
    
    print("\nQuantum Computing Requirements:")
    print(f"  Active space orbitals: {properties.get('active_space_orbitals', 'Unknown')}")
    print(f"  Active space electrons: {properties.get('active_space_electrons', 'Unknown')}")
    print(f"  Qubits needed: {properties.get('hamiltonian_qubits', 'Unknown')}")
    print(f"  Hamiltonian terms: {properties.get('hamiltonian_terms', 'Unknown')}")
    
    print(f"\n{'='*60}")
    print("Hamiltonian generated successfully!")
    print("Ready for VQE simulation on quantum computer.")
    print(f"{'='*60}")
    
    return hamiltonian, properties


if __name__ == "__main__":
    # Process your amoxicillin CSV file with correct path
    csv_file_path = "Database/Quantum_Database/amoxicillin_Nm.csv"
    
    try:
        hamiltonian, properties = process_amoxicillin(csv_file_path)
        
        print("\n=== EXECUTION SUMMARY ===")
        print(f"✓ Successfully processed amoxicillin molecule")
        print(f"✓ Generated {properties['hamiltonian_qubits']}-qubit Hamiltonian")
        print(f"✓ Hamiltonian contains {properties['hamiltonian_terms']} Pauli terms")
        print(f"✓ Ready for quantum simulation!")
        
    except FileNotFoundError:
        print(f"Error: {csv_file_path} file not found")
        print("Available files in Database/Quantum_Database/:")
        try:
            import os
            db_path = "Database/Quantum_Database/"
            if os.path.exists(db_path):
                files = [f for f in os.listdir(db_path) if f.endswith('.csv')]
                for file in files:
                    print(f"  - {file}")
            else:
                print("  Directory not found")
        except:
            print("  Could not list directory contents")
        
    except Exception as e:
        print(f"Error processing molecule: {e}")
        print("Please check your CSV file format and try again")