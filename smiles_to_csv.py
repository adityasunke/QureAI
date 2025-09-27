import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger
import csv
import re

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def smiles_to_csv(smiles, molecule_name):
    """
    Convert SMILES to 3D coordinates and save as CSV.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        molecule_name (str): Name of the molecule (used for filename)
    
    Returns:
        str: Filename of the generated CSV file
    """
    
    print(f"{molecule_name.upper()} - SMILES to 3D Conversion")
    print("=" * 60)
    print(f"Molecule: {molecule_name}")
    print(f"SMILES: {smiles}")
    print()
    
    try:
        # Step 1: Parse SMILES
        print("Step 1: Parsing SMILES...")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES string")
        print("✓ SMILES parsed successfully")
        
        # Step 2: Add explicit hydrogens
        print("\nStep 2: Adding hydrogens...")
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        print(f"✓ Added hydrogens: {num_atoms} total atoms")
        
        # Step 3: Generate 3D coordinates using ETKDG
        print("\nStep 3: Generating 3D coordinates...")
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_result = AllChem.EmbedMolecule(mol, params)
        
        if embed_result == -1:
            raise RuntimeError("Failed to generate 3D coordinates")
        print("✓ 3D coordinates generated using ETKDG method")
        
        # Step 4: Optimize geometry with MMFF94
        print("\nStep 4: Optimizing geometry with MMFF94...")
        opt_result = AllChem.MMFFOptimizeMolecule(mol)
        convergence_status = "converged" if opt_result == 0 else "not fully converged"
        print(f"✓ Geometry optimization completed ({convergence_status})")
        
        # Calculate energy if possible
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol)
            if ff:
                energy = ff.CalcEnergy()
                print(f"✓ MMFF94 Energy: {energy:.2f} kcal/mol")
        except:
            print("  (Energy calculation not available)")
        
        # Extract molecular properties
        print("\nMolecular Properties:")
        print("-" * 30)
        atomic_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        
        print(f"Molecular weight: {mol_weight:.2f} Da")
        print(f"Formal charge: {charge}")
        print(f"Number of atoms: {num_atoms}")
        
        # Count atoms by element
        from collections import Counter
        atom_counts = Counter(atomic_symbols)
        print("Atomic composition:", dict(atom_counts))
        
        # Extract 3D coordinates
        print("\nExtracting 3D coordinates...")
        conf = mol.GetConformer()
        coordinates = []
        
        for atom_idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            coordinates.append([atomic_symbols[atom_idx], pos.x, pos.y, pos.z])
        
        print(f"✓ Extracted coordinates for {len(coordinates)} atoms")
        
        # Generate safe filename from molecule name
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', molecule_name)
        safe_name = safe_name.replace(' ', '_').lower()
        safe_name = re.sub(r'_+', '_', safe_name).strip('_')
        csv_filename = f"{safe_name}_3d.csv"
        
        # Save to CSV file in the requested format: Atom,x,y,z
        print(f"\nSaving coordinates to CSV file: {csv_filename}")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Atom', 'x', 'y', 'z'])
            
            # Write coordinates
            for atom, x, y, z in coordinates:
                writer.writerow([atom, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        
        print(f"✓ CSV file saved successfully!")
        print(f"✓ File contains {len(coordinates)} rows of atomic coordinates")
        
        # Display coordinate preview
        print(f"\n{'='*70}")
        print(f"{molecule_name.upper()} 3D COORDINATES PREVIEW")
        print(f"{'='*70}")
        print("Atom    X         Y         Z")
        print("-" * 35)
        
        # Show first 15 atoms
        for i in range(min(15, len(coordinates))):
            atom, x, y, z = coordinates[i]
            print(f"{atom:2s}   {x:8.4f} {y:8.4f} {z:8.4f}")
        
        if len(coordinates) > 15:
            print(f"... and {len(coordinates)-15} more atoms")
        
        # Show CSV format preview
        print(f"\nCSV Format Preview:")
        print("-" * 40)
        print("Atom,x,y,z")
        for i in range(min(10, len(coordinates))):
            atom, x, y, z = coordinates[i]
            print(f"{atom},{x:.6f},{y:.6f},{z:.6f}")
        if len(coordinates) > 10:
            print("...")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successfully converted {molecule_name}")
        print(f"✓ Generated 3D structure with {num_atoms} atoms")
        print(f"✓ Molecular weight: {mol_weight:.2f} Da")
        print(f"✓ Formal charge: {charge}")
        print(f"✓ Geometry optimization: {convergence_status}")
        print(f"✓ Output saved to: {csv_filename}")
        print(f"✓ CSV format: Atom,x,y,z (as requested)")
        
        return csv_filename
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Example 1: CEFOTAXIME SODIUM
    smiles = input("Enter SMILES: ")
    filename = smiles_to_csv(smiles, "SP")
    if filename:
        print(f"\nSuccess! CSV file created: {filename}")
    else:
        print("\nConversion failed!")