# AI-Based 3D Drug Molecule Generation

AI-powered system for generating candidate drug molecules from protein structures (PDB files) using **Machine Learning, RDKit, Streamlit, and 3D Visualization**.

## Features

✅ Upload protein `.pdb` file
✅ Generate multiple drug candidate molecules
✅ AI-based molecular scoring
✅ Interactive 3D visualization
✅ Molecular property calculation
✅ Download generated molecules as `.pdb`

---

## Technologies Used

* Python
* Streamlit
* RDKit
* BioPython
* Machine Learning (`model.pkl`)
* py3Dmol

---

## Metrics Used

* Molecular Weight (MW)
* LogP
* TPSA
* QED Score
* Predicted pIC50

---

## Project Workflow

PDB File
⬇
Protein Feature Extraction
⬇
Candidate Molecule Generation
⬇
ML Prediction (pIC50)
⬇
Molecule Scoring
⬇
3D Visualization
⬇
Download PDB File

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
streamlit run app.py
```

---

## Example Input

Upload a protein PDB file:

```text
3PTB.pdb
```

---

## Future Scope

* Molecular Docking
* ADMET Prediction
* Deep Learning Molecule Generation
* Real Drug Database Integration

---

## Authors

**Bhuvan Reddy & Team**

