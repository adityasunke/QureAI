import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger
import csv
import re
import os
import pandas as pd

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def create_output_directory(base_name="3D_Coordinates_Output"):
    """Create a new directory for output files."""
    counter = 1
    dir_name = base_name
    
    while os.path.exists(dir_name):
        dir_name = f"{base_name}_{counter}"
        counter += 1
    
    os.makedirs(dir_name)
    print(f"✓ Created output directory: {dir_name}")
    return dir_name

def smiles_to_csv_coordinates(smiles, molecule_name, output_dir):
    """
    Convert SMILES to 3D coordinates and save as CSV.
    
    Args:
        smiles (str): SMILES string representation of the molecule
        molecule_name (str): Name of the molecule (used for filename)
        output_dir (str): Directory to save the output file
    
    Returns:
        tuple: (success_status, filename, error_message)
    """
    
    print(f"\nProcessing: {molecule_name}")
    print("=" * 50)
    print(f"SMILES: {smiles}")
    
    try:
        # Step 1: Parse SMILES
        print("Step 1: Parsing SMILES...")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not parse SMILES string")
        print("✓ SMILES parsed successfully")
        
        # Step 2: Add explicit hydrogens
        print("Step 2: Adding hydrogens...")
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        print(f"✓ Added hydrogens: {num_atoms} total atoms")
        
        # Step 3: Generate 3D coordinates using ETKDG
        print("Step 3: Generating 3D coordinates...")
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_result = AllChem.EmbedMolecule(mol, params)
        
        if embed_result == -1:
            raise RuntimeError("Failed to generate 3D coordinates")
        print("✓ 3D coordinates generated using ETKDG method")
        
        # Step 4: Optimize geometry with MMFF94
        print("Step 4: Optimizing geometry with MMFF94...")
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
        atomic_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        charge = Chem.rdmolops.GetFormalCharge(mol)
        
        print(f"✓ Molecular weight: {mol_weight:.2f} Da")
        print(f"✓ Formal charge: {charge}")
        print(f"✓ Number of atoms: {num_atoms}")
        
        # Extract 3D coordinates
        print("Step 5: Extracting 3D coordinates...")
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
        csv_filename = f"{safe_name}_3d_coordinates.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        
        # Save to CSV file in the requested format: Atom,x,y,z
        print(f"Step 6: Saving to CSV file: {csv_filename}")
        
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Atom', 'x', 'y', 'z'])
            
            # Write coordinates
            for atom, x, y, z in coordinates:
                writer.writerow([atom, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        
        print(f"✓ CSV file saved successfully!")
        print(f"✓ File contains {len(coordinates)} rows of atomic coordinates")
        
        # Show preview of coordinates
        print("\nCoordinate Preview (first 5 atoms):")
        print("Atom    X         Y         Z")
        print("-" * 35)
        for i in range(min(5, len(coordinates))):
            atom, x, y, z = coordinates[i]
            print(f"{atom:2s}   {x:8.4f} {y:8.4f} {z:8.4f}")
        
        if len(coordinates) > 5:
            print(f"... and {len(coordinates)-5} more atoms")
        
        return True, csv_filepath, None
        
    except Exception as e:
        error_msg = f"Error during conversion: {str(e)}"
        print(f"✗ {error_msg}")
        return False, None, error_msg

def process_csv_batch(input_csv_file):
    """
    Process all SMILES from the input CSV file and generate 3D coordinate files.
    
    Args:
        input_csv_file (str): Path to the input CSV file
    """
    
    print("SMILES TO 3D COORDINATES BATCH PROCESSOR")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_directory("generated_Streptococcus_agalactiae_3D_Coords")
    
    # Read the input CSV file
    try:
        df = pd.read_csv(input_csv_file)
        print(f"✓ Loaded {len(df)} compounds from {input_csv_file}")
        print(f"✓ Columns found: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error reading input file: {e}")
        return
    
    # Check if required columns exist
    if 'Smiles' not in df.columns:
        print("✗ Error: 'Smiles' column not found in input file")
        return
    
    if 'Name' not in df.columns:
        print("✗ Error: 'Name' column not found in input file")
        return
    
    # Process each compound
    results = []
    successful_conversions = 0
    failed_conversions = 0
    
    print(f"\nStarting batch processing...")
    print("=" * 60)
    
    for idx, row in df.iterrows():
        smiles = row['Smiles']
        name = row['Name']
        
        print(f"\n[{idx + 1}/{len(df)}] Processing compound: {name}")
        
        success, filepath, error = smiles_to_csv_coordinates(smiles, name, output_dir)
        
        if success:
            successful_conversions += 1
            results.append({
                'Name': name,
                'SMILES': smiles,
                'Status': 'Success',
                'Output_File': os.path.basename(filepath),
                'Error': None
            })
            print(f"✓ Successfully processed {name}")
        else:
            failed_conversions += 1
            results.append({
                'Name': name,
                'SMILES': smiles,
                'Status': 'Failed',
                'Output_File': None,
                'Error': error
            })
            print(f"✗ Failed to process {name}: {error}")
    
    # Create summary report
    summary_file = os.path.join(output_dir, "processing_summary.csv")
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_file, index=False)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total compounds processed: {len(df)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions/len(df)*100):.1f}%")
    print(f"Output directory: {output_dir}")
    print(f"Summary report: {summary_file}")
    
    if successful_conversions > 0:
        print(f"\n✓ {successful_conversions} CSV files with 3D coordinates created!")
        print("Each file contains columns: Atom, x, y, z")
    
    if failed_conversions > 0:
        print(f"\n✗ {failed_conversions} compounds failed to convert")
        print("Check the summary report for error details")
    
    print("\nFiles in output directory:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

# Main execution
if __name__ == "__main__":
    # Process the uploaded CSV file
    input_file = "generated_Streptococcus_agalactiae.csv"
    
    if os.path.exists(input_file):
        process_csv_batch(input_file)
    else:
        print(f"Error: Input file '{input_file}' not found")
        print("Please make sure the CSV file is in the same directory as this script")
        
        # Alternative: ask user for file path
        input_file = input("Enter the path to your CSV file: ").strip()
        if os.path.exists(input_file):
            process_csv_batch(input_file)
        else:
            print(f"Error: File '{input_file}' not found")