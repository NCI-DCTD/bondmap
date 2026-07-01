# BondMap

A Python-based cheminformatics pipeline designed to identify unknown compounds in mixtures by integrating **Infrared (IR) Spectroscopy** with **Mass Spectrometry (MS)** data. Utilizes a multi-output random forest model to predict IR functional groups using SMARTS (SMiles ARbitary Target Specification, https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) as ground truth for IR spectral signals.

## ⚙️ Installation

### Requirements

- Python v 3.11.5 or greater

- Local version of COCONUT library (https://coconut.naturalproducts.net/download) 

- Jupyter notebook (optional)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NCI-DCTD/bondmap.git
   cd bondmap

2. **Install supporting packages:**
   ```bash
   pip install -r requirements.txt

3. **Download COCONUT Library:**

4. **Install jupyter notebook**
   ```bash
   pip install notebook
   jupyter notebook source/bondmap.ipynb
