# 10-708 Probabilistic Graphical Model Project

This repository contains the implementation of a class project for the 10-708 Probabilistic Graphical Model course at Carnegie Mellon University. The goal of this project is to enhance the posterior approximation of NVIDIA's NVAE using a custom three-layer Restricted Boltzmann Machine (RBM) trained on MIMIC-CXR data.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
3. [Scripts and Usage](#scripts-and-usage)
   - [Parsing Radiology Reports](#parsing-radiology-reports)
   - [Training the RBM](#training-the-rbm)
   - [Training NVAE](#training-nvae)
   - [Evaluating Models](#evaluating-models)
4. [File Structure](#file-structure)
5. [Installation](#installation)
6. [Acknowledgements](#acknowledgements)

---

## Project Overview

This project builds on NVIDIA's Nouveau Variational Autoencoder (NVAE) to generate synthetic chest X-rays with enhanced latent representation, integrating radiological findings and anatomy correlations. The pipeline involves:

1. **Parsing MIMIC-CXR reports** to extract structured labels.
2. **Training a custom RBM** for posterior sampling.
3. Using the RBM-enhanced posterior for **training NVAE**.
4. Comparing **NVAE results** with those of a standard Variational Autoencoder (VAE).

---

## Data Preparation

- The project uses **MIMIC-CXR JPEG** images and corresponding radiology reports.
- The data should be placed in the directory: `./file-1024/path/to/study`.
- Each radiology report should be a text file located in a nested structure (`p*/p*/s*.txt`).

---

## Scripts and Usage

### Parsing Radiology Reports

Use `script.sh` to extract structured fields (`INDICATION`, `TECHNIQUE`, `COMPARISON`, `FINDINGS`, `IMPRESSION`, and generic report text) from radiology reports and save them as a CSV.

#### Usage:
```bash
bash script.sh [-v] ./file-1024/path/to/study
```
- **Arguments**:
  - `-v`: Enables verbose mode for detailed logs.
  - `<directory_path>`: Path to the directory containing radiology reports.

- **Output**:
  - `output.csv`: A CSV file containing structured data extracted from all reports. The fields include `INDICATION`, `TECHNIQUE`, `COMPARISON`, `FINDINGS`, `IMPRESSION`, and any generic report text.
  - `script_log.txt`: A log file with details about the processing, including any warnings for missing files or headers.

- **Example Usage**:
  To parse all reports in the directory and log detailed progress:

### Training the RBM

The `threelayerRBM.py` script implements a custom three-layer Restricted Boltzmann Machine (RBM) for posterior sampling. It is trained using the structured data generated by `script.sh` and stored in `output.csv`.

#### Steps:
1. Run the script to preprocess the data and train the RBM:
```bash
python threelayerRBM.py
```

2. Outputs:
  - `rbm_model.pkl`: Trained RBM model.
  - `vectorizer_v.pkl` and `hidden_vectorizers.pkl`: Vectorizers for encoding visible and hidden layer features.

### Training NVAE
Run the modified train.py script to train the NVAE using the RBM-enhanced posterior.
#### Steps:
1. Train NVAE with:
```bash
python train.py --root .
```
2. Outputs:
  - Model checkpoint in `eval-exp` directory.
  - Logs for TensorBoard visualization.


### Evaluating Models
1. Evaluate NVAE using the `NVAE_evaluation.ipynb` notebook.
2. For baseline comparisons, train and evaluate a normal VAE using `normalvae.ipynb`.


## File Structure
.
├── file-1024/path/to/study/   # Directory containing MIMIC-CXR JPEG and reports
├── output.csv                 # Processed radiology reports
├── script.sh                  # Script to parse reports into CSV
├── threelayerRBM.py           # Three-layer RBM implementation
├── train.py                   # NVAE training script
├── NVAE_evaluation.ipynb      # Notebook for NVAE evaluation
├── normalvae.ipynb            # Notebook for baseline VAE training and evaluation
├── eval-exp/                  # Directory for NVAE training outputs
└── README.md                  # Project documentation

## Installation

Clone this repository:
```bash
git clone https://github.com/your-repo/project-name.git
cd project-name
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Download and organize the MIMIC-CXR dataset in the directory structure:
```bash
./file-1024/path/to/study
```
### Acknowledgements
MIMIC-CXR Dataset: Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-CXR.
NVAE Implementation: Based on NVAE by NVIDIA.
CMU 10-708 Course: Probabilistic Graphical Models, Fall 2024.






