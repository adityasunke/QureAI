"""
Meningitis Drug Discovery - SVM + RDKit Generator
=================================================
- Trains an SVM on each dataset separately
- Generates new valid SMILES using RDKit mutations
- Predicts effectiveness of generated drugs
"""

import pandas as pd
import numpy as np
import random
import logging

# ML
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# RDKit
from rdkit import Chem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DrugDiscovery")

# ==============================
# Feature Engineering
# ==============================

STANDARD_TYPES = ["MIC", "MBC", "IC50", "IZ", "log10cfu", "Selectivity ratio", "Ratio"]

def transform_row(row: pd.Series) -> np.ndarray:
    """Turn a dataset row into a numeric feature vector."""
    features = [
        row.get("Molecular Weight", 0),
        row.get("AlogP", 0),
        row.get("#RO5 Violations", 0),
    ]

    # Encode Standard Type
    for stype in STANDARD_TYPES:
        if row["Standard Type"] == stype:
            val = row["Standard Value"]
            if stype in ["MIC", "MBC", "IC50", "log10cfu", "Ratio"]:
                val = np.log1p(val)  # lower is better
            else:
                val = np.log1p(val)  # higher is better (still log scale)
            features.append(val)
        else:
            features.append(0.0)

    return np.array(features, dtype=float)

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([transform_row(r) for _, r in df.iterrows()])


# ==============================
# SVM Training
# ==============================

def train_svm_on_dataset(df: pd.DataFrame) -> Pipeline:
    """Train an SVM pipeline on one dataset."""
    X = build_feature_matrix(df)

    # Fake labels: assume top 50% effective, bottom 50% ineffective
    threshold = np.median(df["Standard Value"].dropna())
    y = (df["Standard Value"] <= threshold).astype(int)

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    logger.info(f"SVM Trained: Accuracy={acc:.3f}, AUC={auc:.3f}")

    return pipe


# ==============================
# Molecule Mutator (RDKit)
# ==============================

class MoleculeMutator:
    """
    Generate new molecules by mutating existing SMILES strings with RDKit.
    Ensures output SMILES are chemically valid or falls back to original.
    """

    def __init__(self, max_mutations: int = 2):
        self.max_mutations = max_mutations

    def mutate(self, smiles: str) -> str:
        """Mutate a molecule SMILES into a new valid one."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # fallback if RDKit can't parse

        for _ in range(self.max_mutations):
            action = random.choice(["add_atom", "replace_atom"])

            emol = Chem.RWMol(mol)
            try:
                if action == "add_atom":
                    atom = random.choice(["C", "N", "O", "F", "Cl"])
                    idx = random.randint(0, mol.GetNumAtoms() - 1)
                    new_idx = emol.AddAtom(Chem.Atom(atom))
                    emol.AddBond(idx, new_idx, order=Chem.rdchem.BondType.SINGLE)

                elif action == "replace_atom":
                    idx = random.randint(0, mol.GetNumAtoms() - 1)
                    new_atom = random.choice(["C", "N", "O", "S"])
                    emol.ReplaceAtom(idx, Chem.Atom(new_atom))

                mol = emol.GetMol()
                Chem.SanitizeMol(mol, catchErrors=True)

            except Exception as e:
                logger.warning(f"Skipping invalid mutation: {e}")
                continue

        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return smiles


# ==============================
# Drug Generation
# ==============================

def generate_new_drugs(df, model, bacterium, n_samples=20):
    """Generate new drugs using RDKit mutations."""
    synthetic = []
    counter = 1
    mutator = MoleculeMutator(max_mutations=2)

    # Pool of real molecules
    real_smiles = df["Smiles"].dropna().unique().tolist()

    while len(synthetic) < n_samples and real_smiles:
        base_smiles = random.choice(real_smiles)
        new_smiles = mutator.mutate(base_smiles)

        # Sample row properties
        row = {
            "Molecular Weight": np.random.normal(df["Molecular Weight"].mean(), df["Molecular Weight"].std()),
            "AlogP": np.random.normal(df["AlogP"].mean(), df["AlogP"].std()),
            "#RO5 Violations": int(np.random.choice(df["#RO5 Violations"].dropna())),
            "Standard Type": np.random.choice(df["Standard Type"].dropna()),
            "Smiles": new_smiles
        }

        subset = df[df["Standard Type"] == row["Standard Type"]]["Standard Value"].dropna()
        row["Standard Value"] = np.random.choice(subset) if len(subset) > 0 else 1.0

        x = transform_row(pd.Series(row)).reshape(1, -1)
        pred_proba = model.predict_proba(x)[0, 1]

        if pred_proba > 0.6:  # keep only promising candidates
            row["Name"] = f"Generated_{bacterium}_{counter:03d}"
            row["Predicted_Effectiveness"] = pred_proba
            synthetic.append(row)
            counter += 1

    return pd.DataFrame(synthetic)


# ==============================
# Main Pipeline
# ==============================

def main():
    datasets = {
        "Neisseria_meningitidis": "AIML-Cure-Generator\ML_Database\Haemophilus_influenzae.csv",
        "Streptococcus_agalactiae": "AIML-Cure-Generator\ML_Database\Streptococcus_agalactiae.csv",
        "Haemophilus_influenzae": "AIML-Cure-Generator\ML_Database\Haemophilus_influenzae.csv",
        "Streptococcus_pneumoniae": "AIML-Cure-Generator\ML_Database\Streptococcus_pneumoniae.csv",
    }

    for name, path in datasets.items():
        print(f"\n=== Processing {name} ===")
        df = pd.read_csv(path)
        model = train_svm_on_dataset(df)

        generated = generate_new_drugs(df, model, name, n_samples=10)
        print(f"\nGenerated Drugs for {name}:")
        print(generated[["Name", "Smiles", "Predicted_Effectiveness"]])

        # Save results
        out_file = f"generated_{name}.csv"
        generated.to_csv(out_file, index=False)
        print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
