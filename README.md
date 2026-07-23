# BondMap

A Python-based cheminformatics pipeline designed to identify unknown compounds in mixtures by integrating **Infrared (IR) Spectroscopy** with **Mass Spectrometry (MS)** data. Utilizes a multi-output random forest model to predict IR functional groups using SMARTS (SMiles ARbitary Target Specification, https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) as ground truth for IR spectral signals.

## Prerequisites
- Python 3.11 or later
- Local copy of the COCONUT library (https://coconut.naturalproducts.net/download)
  (developed/tested against the May 2026 CSV release but newer versions work)

## Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/NCI-DCTD/bondmap.git
   cd bondmap
```

2. **Create and activate a virtual environment:**
```bash
   # 2.1 Create a virtual environment for installing requirements
   python -m venv venv           # or python3 depending on environment

   # 2.2 Activate the environment for use
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
```

3. **Install dependencies:**
```bash
   # make sure virtual environment is activated first
   pip install -r requirements.txt
```

4. **Download the COCONUT library** unzip and place the CSV file in the `stored/` folder at the repository root:
```
   bondmap/
   ├── source/
   │   └── bondmap.ipynb
   └── stored/
       └── coconut_csv-05-2026.csv
```
>If your file is named differently or stored elsewhere, update the path in the notebook's library-loading cell:
   ```python
      library_df = load_smiles_library("../stored/coconut_csv-05-2026.csv")
   ```

5. **(Optional) Install Jupyter, if not already installed:**
```bash
   pip install notebook
   jupyter notebook source/bondmap.ipynb
```

## Verifying your setup
Run the COCONUT-loading cell first to verify proper installation.