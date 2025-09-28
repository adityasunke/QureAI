"""
Molecular Hamiltonian Generator
Generates quantum Hamiltonians from CSV molecular coordinates
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Dict
import warnings
from pyscf import gto, scf, mcscf, ao2mo
from qiskit.quantum_info import SparsePauliOp


def parse_csv_geometry(csv_path: str) -> Tuple[List, Dict]:
    # Parse molecular geometry from CSV file.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Reading molecular geometry from: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Validate required columns
    required_cols = ['atom', 'x', 'y', 'z']
    
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


def jordan_wigner_transformation(n_qubits: int) -> Tuple[List[SparsePauliOp], List[SparsePauliOp]]:
    # Generate creation and annihilation operators using Jordan-Wigner transformation.

    creation_ops = []
    
    for p in range(n_qubits):
        pauli_string = ["I"] * n_qubits
        
        # Add Z operators for anticommutation
        for k in range(p):
            pauli_string[k] = "Z"
        
        # Creation operator
        pauli_string[p] = "X"
        x_string = "".join(reversed(pauli_string))
        
        pauli_string[p] = "Y"
        y_string = "".join(reversed(pauli_string))
        
        c_p = SparsePauliOp.from_list([
            (x_string, 0.5),
            (y_string, 0.5j)
        ])
        creation_ops.append(c_p)
    
    # Annihilation operators
    annihilation_ops = [c_op.adjoint() for c_op in creation_ops]
    
    return creation_ops, annihilation_ops


def cholesky_decomposition(V: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, int]:
    # Cholesky decomposition for two-electron integrals.

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


def build_molecular_hamiltonian_from_integrals(ecore: float, h1e: np.ndarray, h2e: np.ndarray) -> SparsePauliOp:
    # Build molecular Hamiltonian from integral data.

    ncas = h1e.shape[0]
    n_qubits = 2 * ncas
    
    print(f"Building Hamiltonian: {ncas} orbitals, {n_qubits} qubits")
    
    # Get Jordan-Wigner operators
    C, D = jordan_wigner_transformation(n_qubits)
    
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
    L_chol, ng = cholesky_decomposition(h2e, eps=1e-6)
    
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
    
    print(f"Hamiltonian built: {len(H)} terms")
    
    return H


def determine_active_space(total_orbitals: int, total_electrons: int, spin: int = 0) -> Tuple[int, Tuple[int, int]]:
    # Automatically determine active space size (kept small for efficiency).

    # Conservative active space selection
    if total_orbitals > 50:
        ncas = 4  # Very small for large molecules
    elif total_orbitals > 30:
        ncas = 6
    elif total_orbitals > 15:
        ncas = 8
    else:
        ncas = min(6, total_orbitals)
    
    # Ensure we don't exceed system limits
    ncas = min(ncas, total_orbitals)
    
    # Distribute electrons in active space
    if spin == 0:
        active_electrons = min(ncas * 2, total_electrons)
        nelecas = (active_electrons // 2, active_electrons // 2)
    else:
        alpha_excess = spin
        active_electrons = min(ncas * 2, total_electrons)
        alpha_electrons = (active_electrons + alpha_excess) // 2
        beta_electrons = active_electrons - alpha_electrons
        nelecas = (alpha_electrons, beta_electrons)
    
    print(f"Active space: {ncas} orbitals, {nelecas} electrons, {2*ncas} qubits")
    
    return ncas, nelecas


def generate_hamiltonian(csv_path: str) -> Tuple[SparsePauliOp, Dict]:
    # Main function: Generate molecular Hamiltonian from CSV coordinates.

    print("=" * 60)
    print("GENERATING MOLECULAR HAMILTONIAN")
    print("=" * 60)
    
    # Parse geometry
    atom_list, csv_properties = parse_csv_geometry(csv_path)
    
    # Build molecule
    print("Building PySCF molecule...")
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=atom_list,
        basis="sto-3g",
        charge=0,
        spin=0
    )
    
    print(f"Molecule properties:")
    print(f"  Total electrons: {mol.nelectron}")
    print(f"  Total orbitals: {mol.nao_nr()}")
    
    # Run SCF calculation
    print("Running SCF calculation...")
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    
    mf.max_cycle = 10
    mf.conv_tol = 1e-6
    energy = mf.kernel()
    
    if not mf.converged:
        warnings.warn("SCF did not converge!")
    
    print(f"SCF energy: {energy:.6f} Hartree")
    
    # Determine active space
    ncas, nelecas = determine_active_space(mol.nao_nr(), mol.nelectron, mol.spin)
    
    # Setup CASCI calculation
    print("Setting up active space calculation...")
    cas = mcscf.CASCI(mf, ncas=ncas, nelecas=nelecas)
    
    # Select orbitals around HOMO-LUMO gap
    homo_idx = mol.nelectron // 2 - 1
    start_idx = max(0, homo_idx - ncas // 2 + 1)
    active_orbitals = list(range(start_idx, start_idx + ncas))
    
    mo = cas.sort_mo(active_orbitals, base=0)
    cas_energy = cas.kernel(mo)
    
    print(f"CASCI energy: {cas.e_tot:.6f} Hartree")
    
    # Extract Hamiltonian components
    h1e, ecore = cas.get_h1eff()
    h2e = ao2mo.restore(1, cas.get_h2eff(), cas.ncas)
    
    # Build quantum Hamiltonian
    hamiltonian = build_molecular_hamiltonian_from_integrals(ecore, h1e, h2e)
    
    # Compile properties
    properties = {
        **csv_properties,
        "total_electrons": mol.nelectron,
        "total_orbitals": mol.nao_nr(),
        "active_space_orbitals": ncas,
        "active_space_electrons": nelecas,
        "scf_energy": energy,
        "casci_energy": cas.e_tot,
        "num_qubits": hamiltonian.num_qubits,
        "hamiltonian_terms": len(hamiltonian)
    }
    
    print("=" * 60)
    print("HAMILTONIAN GENERATION COMPLETE")
    print(f"Qubits: {hamiltonian.num_qubits}")
    print(f"Pauli terms: {len(hamiltonian)}")
    print("=" * 60)
    
    return hamiltonian, properties


if __name__ == "__main__":
    # Test with a sample CSV file
    test_csv = "test_molecule.csv"
    
    if os.path.exists(test_csv):
        hamiltonian, properties = generate_hamiltonian(test_csv)
        print(f"Generated Hamiltonian for {properties['composition']}")
    else:
        print(f"Test file {test_csv} not found")
        print("Create a CSV with columns: Atom,x,y,z to test this module")