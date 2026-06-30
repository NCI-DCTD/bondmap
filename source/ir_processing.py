"""
Spectroscopy Data Processing Module

This module provides classes for loading, processing, and standardizing 
spectroscopic data (Bruker OPUS and JCAMP-DX formats), as well as managing 
training datasets for machine learning applications.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Third-party parsers
from brukeropusreader import read_file
from jcamp import jcamp_readfile

# Local imports
from smiles_processing import smi_to_canon

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── 1. DEFINED TRAINING DATA CLASS ──────────────────────────────────────

class Spectrum:
    """
    Represents a single spectroscopic reading. Handles loading from file, 
    baseline correction, and standardization to absorbance.
    """

    def __init__(self, spec_path: str | Path, sample_id: str, smiles: Optional[str] = None):
        """
        Initialize the Spectrum object and trigger data processing.

        Args:
            spec_path (str | Path): Path to the spectrum file (.0 or .jdx).
            sample_id (str): Unique identifier for the sample.
            smiles (str, optional): SMILES string of the molecule. Defaults to None.
        """
        self.spec_path = Path(spec_path)
        self.sample_id = sample_id
        self.smiles = smiles

        # Masked wavenumbers and raw readings from file
        _wn, self.abs_orig = self._load_spectrum()

        # Readings transformed to absorbance if necessary
        self.wn, self.abs_nobaseline = self._standardise_to_absorbance(_wn)

        # Baseline corrected absorbance values used for training and predicting
        self.absorbance = self._process_spectrum()

    # ── 2. SPECTRUM LOADING & PROCESSING ─────────────────────────────────────────

    def _load_spectrum(self, low_cm: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the spectrum data from the file and applies a wavenumber mask.

        Args:
            low_cm (int): The lower bound for the wavenumber mask.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Masked wavenumbers and absorbance values.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        match self.spec_path.suffix.lower():
            case ".0":
                data = read_file(self.spec_path)
                ab = data["AB"]
                ab_param = data["AB Data Parameter"]
                fxv = ab_param["FXV"]
                lxv = ab_param["LXV"]
                wn = np.linspace(fxv, lxv, len(ab))
            case ".jdx":
                data = jcamp_readfile(self.spec_path)
                wn = data['x']
                ab = data['y']
            case _:
                raise ValueError(f"File format '{self.spec_path.suffix}' not supported.")

        mask = wn >= low_cm
        return wn[mask], ab[mask]

    def _process_spectrum(self) -> np.ndarray:
        """
        Applies asymmetric least squares baseline correction.

        Returns:
            np.ndarray: Baseline-corrected absorbance values.
        """
        baseline = self._als_baseline()
        ab_corrected = self.abs_nobaseline - baseline
        return ab_corrected

    def _standardise_to_absorbance(self, _wn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardizes raw readings to normalized absorbance values.

        Args:
            _wn (np.ndarray): Array of wavenumbers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Adjusted wavenumbers and normalized absorbance.
            
        Raises:
            ValueError: If the extracted wavenumber array is empty.
        """
        s = np.array(self.abs_orig, dtype=float)
        wn = np.array(_wn, dtype=float)

        if len(wn) == 0:
            raise ValueError("No spectra extracted. Zero length wavenumber array.")

        # Reverse the wavenumbers and adjust raw readings if necessary
        if wn[0] > wn[-1]:
            wn, s = wn[::-1], s[::-1]

        baseline = np.percentile(s, 95)

        # --- Format Detection ---
        if baseline > 10:
            # Clearly percentage transmittance (0-100)
            s = s / 100.0
            is_transmittance = True
        elif baseline > 0.5:
            # Fractional transmittance (0-1), possibly with baseline drift above 1.0
            low = np.percentile(s, 5)
            high = np.percentile(s, 95)
            spread = high - low
            
            # TODO: Review this logic. It was hardcoded to True in the original code.
            # is_transmittance = (spread > 0.3) and (low < 0.5)
            is_transmittance = True 
        else:
            is_transmittance = False

        if is_transmittance:
            # Normalise baseline to 1.0 before inverting to correct drift
            s = s / baseline
            s = -s + 1.0

        # Min-Max Normalization
        normalized = (s - np.min(s)) / (np.max(s) - np.min(s))
        return wn, normalized

    def _als_baseline(self, lam: float = 1e5, p: float = 0.001, n_iter: int = 20) -> np.ndarray:
        """
        Asymmetric least squares baseline correction (Eilers & Boelens 2005).

        Args:
            lam (float): Smoothness parameter. Larger values result in a stiffer baseline.
                         1e4–1e5 better for IR with broad OH bands.
            p (float): Asymmetry parameter. Fraction of weight given to points ABOVE baseline.
                       0.001 strongly penalizes upward excursions (good for absorbance).
            n_iter (int): Number of iterations to ensure convergence on broad features.

        Returns:
            np.ndarray: Calculated baseline.
        """
        L = len(self.abs_nobaseline)
        D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
        w = np.ones(L)
        
        for _ in range(n_iter):
            W = diags(w, 0)
            Z = spsolve(W + lam * D.T @ D, w * self.abs_nobaseline)
            w = np.where(self.abs_nobaseline > Z, p, 1 - p)
            
        return Z


class TrainingData:
    """
    Manages a dataset of Spectrum objects loaded from a CSV index.
    """

    def __init__(self, csv_path: str | Path, spec_root: str | Path):
        """
        Initialize the TrainingData object and load all spectra.

        Args:
            csv_path (str | Path): Path to the CSV containing filenames and SMILES.
            spec_root (str | Path): Root directory containing the spectrum files.
        """
        self.csv_path = Path(csv_path)
        self.spectra_dir = Path(spec_root)

        # Read dataset index
        df = pd.read_csv(self.csv_path, encoding="latin-1")
        self.records = [row.to_dict() for _, row in df.iterrows()]
        
        # Load spectra
        self.spectra = self._format_data()

    def _format_data(self) -> List[Spectrum]:
        """
        Iterates through the dataset index, instantiating Spectrum objects.

        Returns:
            List[Spectrum]: A list of loaded and processed Spectrum objects.
        """
        spectra = []
        for rec in self.records:
            smiles = rec["smiles"]
            fpath = self.spectra_dir / rec["filename"]
            
            try:
                spectra.append(
                    Spectrum(
                        spec_path=fpath,
                        smiles=smi_to_canon(smiles),
                        sample_id=rec['filename']
                    )
                )
            except Exception as e:
                logging.warning(f"Could not load {fpath}: {e}")
                continue
                
        return spectra

    def size(self) -> int:
        """
        Returns the number of successfully loaded spectra.

        Returns:
            int: Number of spectra.
        """
        return len(self.spectra)

