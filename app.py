import streamlit as st
import random
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED
from Bio.PDB import PDBParser
import streamlit.components.v1 as components

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="3D Drug Molecule Generation",
    layout="wide"
)

# ========= LOAD MODEL =========
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ========= SMILES POOL =========
SMILES_LIST = [
    "CCO","CCN","CCC","CCCN","CCCO","CCCl","CCBr",
    "c1ccccc1","c1ccncc1","c1ccccc1O","c1ccccc1N",
    "CC(=O)O","CC(=O)N","CC(C)O","CC(C)N",
    "C1CCCCC1","C1=CC=CC=C1","C1CCNCC1",
    "CCOC","CCS","COC","COCC","CN(C)C",
    "CCN(CC)CC","CC(C)CC","CC(C)CO",
    "CC(C)C(=O)O","CNC","COCO",
    "CCOCC","CCNCO","CCOCN",
    "CC(C)(C)O","CC(C)CN","CC(C)CCO",
    "CC(C)=O","CC(C)C","CCCC","CCCCC",
    "CCCCCC","CCOCCC","CCNCCC",
    "COCCO","CCCOC","CCNCOC",
    "CC(C)OC","CC(C)NC","CCSC",
    "CC(C)Cl","CC(C)Br","COCN",
    "CCOC(=O)C","CCNC(=O)C",
    "c1ccccc1C","c1ccccc1Cl",
    "c1ccccc1Br","c1ccncc1C",
    "CC1=CC=CC=C1","CCC1CCCCC1",
    "CCOCN","CCNCN","CCOCO",
    "CCSCC","COCOC","CNCCN",
    "CC(C)(C)N","CC(C)(C)C",
    "CC(C)(C)CO","CC(C)(C)CN",
    "CC(C)OCC","CC(C)NCC",
    "CCOCCl","CCOCBr",
    "CCN(CC)C","CCN(C)CC",
    "CC(C)CCC","CC(C)CCCC",
    "CC(C)CCN","CC(C)CCCl",
    "CC(C)CCBr","CCCCO",
    "CCCCN","CCCCCl","CCCCBr"
]

# ========= EXTRACT PROTEIN =========
def extract_protein_feature(uploaded_file):

    temp_path = "temp_protein.pdb"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(
        "protein",
        temp_path
    )

    atom_count = 0

    for model_ in structure:
        for chain in model_:
            for residue in chain:
                for atom in residue:
                    atom_count += 1

    return atom_count, temp_path


# ========= FEATURES =========
def get_features(mol):

    return [[
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.FractionCSP3(mol)
    ]]


# ========= GENERATE MOLECULES =========
def generate_molecules(
    protein_size,
    num_molecules
):

    generated = []
    used_smiles = set()

    for mol_num in range(num_molecules):

        candidates = []

        for _ in range(100):

            smiles = random.choice(
                SMILES_LIST
            )

            if smiles in used_smiles:
                continue

            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:
                continue

            mol = Chem.AddHs(mol)

            try:
                AllChem.EmbedMolecule(mol)
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=50
                )

            except:
                continue

            features = get_features(mol)

            predicted_pic50 = (
                model.predict(
                    features
                )[0]
            )

            qed_score = QED.qed(mol)

            final_score = (
                predicted_pic50
                + (0.0005 * protein_size)
                + (0.5 * qed_score)
            )

            candidates.append(
                (
                    final_score,
                    smiles,
                    mol
                )
            )

        candidates.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        selected = random.choice(
            candidates[:20]
        )

        best_smiles = selected[1]
        best_mol = selected[2]

        used_smiles.add(
            best_smiles
        )

        ligand_path = (
            f"molecule_"
            f"{mol_num+1}.pdb"
        )

        Chem.MolToPDBFile(
            best_mol,
            ligand_path
        )

        generated.append({
            "path":
            ligand_path,
            "smiles":
            best_smiles,
            "qed":
            QED.qed(best_mol),
            "logp":
            Descriptors.MolLogP(best_mol),
            "mw":
            Descriptors.MolWt(best_mol),
            "tpsa":
            Descriptors.TPSA(best_mol)
        })

    return generated


# ========= 3D VIEW =========
def show_3d(
    protein_pdb,
    ligand_pdb
):

    html = f"""
    <div id="viewer"
    style="width:100%;
    height:500px;"></div>

    <script src=
    "https://3dmol.org/build/3Dmol-min.js">
    </script>

    <script>

    let viewer =
    $3Dmol.createViewer(
        "viewer",
        {{
            backgroundColor:
            "white"
        }}
    );

    viewer.addModel(
        `{protein_pdb}`,
        "pdb"
    );

    viewer.setStyle(
        {{model:0}},
        {{
            cartoon:
            {{
                color:
                'lightgray'
            }}
        }}
    );

    viewer.addModel(
        `{ligand_pdb}`,
        "pdb"
    );

    viewer.setStyle(
        {{model:1}},
        {{
            stick:
            {{
                radius:0.35,
                colorscheme:
                'greenCarbon'
            }},
            sphere:
            {{
                radius:0.45
            }}
        }}
    );

    viewer.zoomTo({{model:1}});
    viewer.render();

    </script>
    """

    components.html(
        html,
        height=500
    )


# ========= UI =========
st.title(
    "🧬 3D Drug Molecule Generation"
)

uploaded_file = st.file_uploader(
    "Upload PDB File",
    type=["pdb"]
)

num_molecules = st.number_input(
    "Number of Molecules",
    min_value=1,
    max_value=20,
    value=5
)

if st.button(
    "🚀 Generate Molecules"
):

    if uploaded_file is None:
        st.error(
            "Upload a PDB file first"
        )

    else:

        with st.spinner(
            "Generating molecules..."
        ):

            (
                protein_size,
                protein_path
            ) = extract_protein_feature(
                uploaded_file
            )

            generated = (
                generate_molecules(
                    protein_size,
                    num_molecules
                )
            )

            with open(
                protein_path,
                "r"
            ) as f:

                protein_data = (
                    f.read()
                )

            for i, mol in enumerate(
                generated
            ):

                st.markdown("---")

                st.subheader(
                    f"🧪 Molecule {i+1}"
                )

                st.write(
                    f"SMILES: {mol['smiles']}"
                )

                with open(
                    mol["path"],
                    "r"
                ) as f:

                    ligand_data = f.read()

                show_3d(
                    protein_data,
                    ligand_data
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "QED",
                        round(
                            mol["qed"],
                            3
                        )
                    )
                    st.metric(
                        "LogP",
                        round(
                            mol["logp"],
                            3
                        )
                    )

                with col2:
                    st.metric(
                        "MW",
                        round(
                            mol["mw"],
                            2
                        )
                    )
                    st.metric(
                        "TPSA",
                        round(
                            mol["tpsa"],
                            2
                        )
                    )

                with open(
                    mol["path"],
                    "rb"
                ) as file:

                    st.download_button(
                        f"⬇ Download Molecule {i+1}",
                        data=file,
                        file_name=
                        f"molecule_{i+1}.pdb"
                    )

st.caption(
    "RDKit + Streamlit + AI"
)