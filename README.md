# Qure AI  
**Harnessing Machine Learning and Quantum Computing to fight Meningitis**
---
[Link to Portfolio](https://qure-ai.vercel.app/)
---

## Project Overview
**Qure AI** is a hybrid **AI + Quantum** pipeline designed to discover and validate potential drug molecules targeting the bacteria that cause **meningitis**.  

- **AI/ML Component**: A machine learning model (Support Vector Machine) is trained on molecular datasets (sourced from **ChEMBL**) to identify effective drug candidates. We also propose **novel SMILES strings** as potential cures.  
- **Quantum Component**: The **Variational Quantum Eigensolver (VQE)** algorithm is applied to calculate molecular ground-state energies. Lower energies indicate greater molecular stability, providing a validation step for both existing and AI-generated molecules.  

This project demonstrates the **synergy of machine learning and quantum simulation** for next-generation drug discovery.

---

## Motivation
Meningitis remains a **critical global health burden**:  
- Every year, **236,000 people die** of meningitis.  
- Over **2 million new cases** are diagnosed annually.  
- That’s **1 death every 2 minutes** and **1 diagnosis every 15 seconds**.  
- **1 in 5 survivors** live with permanent disabilities (brain damage, limb amputations, kidney failure).  
- Within **1–2 years**, the majority of antibodies decrease rapidly.  

By integrating AI and quantum computing, **Qure AI** aims to accelerate drug discovery and validation, potentially reducing timelines and costs compared to traditional lab methods.

---

## Target Bacteria
We studied molecules that attack **four major bacterial pathogens** responsible for meningitis:  
- *Neisseria meningitidis*  
- *Streptococcus pneumoniae*  
- *Haemophilus influenzae*  
- *Streptococcus agalactiae*  

---

## Dataset
- **Source**: [ChEMBL Database](https://www.ebi.ac.uk/chembl/)  
- **Features used**:  
  - SMILES strings  
  - Molecular weight  
  - MIC (Minimum Inhibitory Concentration)  
  - MBC (Minimum Bactericidal Concentration)  
  - Selective Ratios  
  - IZ (Inhibition Zone)  
  - IC50 values  
  - Log<sub>10</sub>CFU  
  - AlogP  
  - Lipinski’s Rule of 5  

---

## Machine Learning Component
- **Model Used**: Support Vector Machine (SVM)  
- **Objective**: Classify molecules as effective/ineffective against target bacteria  
- **Generative AI**: Produce **novel SMILES molecules** with drug-like properties  

---

## Quantum Validation (VQE)
- **Framework**: Qiskit + PySCF  
- **Process**:  
  1. Convert molecules into Hamiltonians
  2. Choose an appropriate Ansatz (EfficientSU2)
  3. Implement a classical optimizer (COBYLA)
  4. Run **Variational Quantum Eigensolver (VQE)**  
  5. Compare ground-state energies in Hartree (1 Hartree = 27.2 eV)
- **Interpretation**:  
  - **Lower energy → more stable molecule**  
  - Validates both **existing** and **AI-generated** molecules  

---

## Pipeline Workflow
```mermaid
flowchart TD
    A[📂 Clean Dataset] --> B[🤖 Train SVM Model on Molecular Features]
    B --> C[⚛️ Run VQE Simulation for Quantum Validation]
    C --> D[🧬 Generate Novel Molecules via Machine Learning]
    D --> E[📊 Compare Novel vs Existing Molecules by Calculating Energy Levels]
```
---

## Tech Stack
Python, HTML, CSS, Qiskit, PySCF, Pandas, RDKit, NumPy, scikit-learn, Matplotlib, three.js, chart.js, vanta.js

---

## Results
The VQE quantum simulation results demonstrate successful energy convergence for both existing and AI-generated molecules:
<p align="center">
  <img src="Quantum/Outputs/cefotaxime_sodium_Sp_vqe_results.png" alt="Existing Molecule (Cefotaxime sodium)" width="420"/>
  <img src="Quantum/Outputs/generated_streptococcus_pneumoniae_001_3d_coordinates_vqe_results.png" alt="AI-Generated Molecule" width="420"/>
</p>
<p align="center">
  <em><strong>(a) Existing Molecule (Cefotaxime sodium)</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>(b) AI-Generated Molecule</strong></em>
</p>

### Key Findings
- The generated molecule is **1.98x** more stable than existing molecule
---

## References
**Quantum Computing & VQE:**
- [Hamiltonian Construction](https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/hamiltonian-construction)
- [Ansatz](https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/ansatz)
- [Classical Optimizers](https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/classical-optimizers)
- [Ground State](https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/ground-state)

**Machine Learning & ChEMBL:**
- [pChEMBL Value Definitions](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0297-4)

**Meningitis Statistics & General Information:**
- [Global Meningitis Burden](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(23)00195-3/fulltext)
- [Meningitis Statistics](https://nmaus.org/nma-disease-prevention-information/statistics-and-disease-facts/)
- [CDC Vaccine Information](https://www.cdc.gov/vaccines/vpd/mening/hcp/about-vaccine.html)
- [Antibiotic Resistance Data](https://biomedpharmajournal.org/vol16no1/antibiotic-resistance-of-streptococcus-pneumoniae-neisseria-meningitidis-haemophilus-influenzae-and-staphylococcus-aureus-in-morocco-national-data-meta-analysis/#:~:text=pneumoniae%20to%20penicillin%20G%20increased,16.5%25%2C%20erythromycin%2014.6%25)

**Target Bacteria (ChEMBL):**
- [Neisseria meningitidis](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL347)
- [Streptococcus pneumoniae](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL614431)
- [Haemophilus influenzae](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL355)
- [Streptococcus agalactiae](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL614622)

**Parameter Research:**
- [MIC (Minimum Inhibitory Concentration)](https://journals.asm.org/doi/abs/10.1128/aac.32.8.1131)
- [IC50](https://www.ingentaconnect.com/content/ben/ctmc/2012/00000012/00000011/art00008)
- [MBC (Minimum Bactericidal Concentration)](https://www.sciencedirect.com/science/article/pii/S0732889313005956)
- [AlogP](https://pubs.acs.org/doi/full/10.1021/ci034134i)
