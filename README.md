# BsiteP: Protein-ligand binding site prediction
BsiteP is a method for predicting the binding sites of proteins for ligands.


## Installation

```bash
conda env create -f environment.yaml
conda activate BsiteP
```

## Usage

### Running the Prediction

You can use the predict.ipynb notebook to run the protein-ligand binding site prediction.
Open the Jupyter lab:

```bash
jupyter-lab predict.ipynb
```

### Evaluating Model Performance
To evaluate the performance of the trained model, open the `DCCDCA.ipynb` notebook. This notebook provides methods to assess model performance using the following metrics:

```bash
jupyter-lab DCCDCA.ipynb
```

- **DCC (Distance from site Centre to ligand centre)**: Measures the distance between the predicted binding site center and the ligand center.
- **DCA (Distance from site Centre to ligand atom)**: Measures the distance between the predicted binding site center and individual ligand atoms.

Additionally, you can visualize the predicted binding sites using `py3Dmol`, a powerful tool for 3D molecular visualization. This allows for detailed visual inspection of the predicted binding sites and their proximity to the ligands.