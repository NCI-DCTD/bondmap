"""
SMILES Processing and Standardization Module

This module provides utility functions to standardize SMILES strings, calculate 
exact and average molecular weights, and identify functional groups (FGs) via 
SMARTS matching with the RDKit chemistry engine.
"""

import logging
from typing import Tuple, Optional, FrozenSet

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MolStandardize, rdMolDescriptors, Descriptors

# Local imports
from fg_smarts_map import FG_IR_MAP

# Suppress standard warnings from RDKit
RDLogger.DisableLog("rdApp.*")

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Pre-compile SMARTS patterns once at import time for maximum efficiency
_COMPILED_SMARTS = {
    label: Chem.MolFromSmarts(smarts)
    for label, smarts, _, _ in FG_IR_MAP
}


def clean_fragment(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    """
    Standardizes a chemical structure by removing hydrogens, disconnecting metal 
    atoms, reionizing, and returning the largest organic fragment (parent).

    Args:
        mol (Optional[Chem.Mol]): RDKit molecule object to be cleaned.

    Returns:
        Optional[Chem.Mol]: The standardized parent molecule, or None if input is invalid.
    """
    if mol is None:
        return None
        
    try:
        # Standardize, remove salt ions, and normalize charge states
        clean_mol = MolStandardize.rdMolStandardize.Cleanup(mol) 
        
        # Isolate the main organic component (ignores salt adducts, metals, water)
        parent_clean_mol = MolStandardize.rdMolStandardize.FragmentParent(clean_mol)
        return parent_clean_mol
    except Exception as e:
        logging.error(f"Failed standardizing molecule fragment: {e}")
        return None


def smi_to_canon(smiles: str) -> Optional[str]:
    """
    Converts a raw SMILES string into its standard canonical form.

    Args:
        smiles (str): Raw input SMILES string.

    Returns:
        Optional[str]: Canonicalized and cleaned SMILES string.

    Raises:
        TypeError: If the input SMILES is not a string.
        ValueError: If the input is empty or RDKit fails to parse the structure.
    """
    if not isinstance(smiles, str):
        raise TypeError(f"Expected str, got {type(smiles).__name__}")
    if not smiles.strip():
        raise ValueError("SMILES string must not be empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")

    mol = clean_fragment(mol)
    if mol is None:
        raise ValueError(f"clean_fragment returned None for: {smiles!r}")

    result = Chem.MolToSmiles(mol)
    if not result:
        raise ValueError(f"MolToSmiles produced an empty result for: {smiles!r}")

    return result


def exact_mass_from_smiles(smiles: str) -> Tuple[float, int]:
    """
    Calculates the monoisotopic exact mass and net charge from a SMILES string.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        Tuple[float, int]: A tuple containing:
                           - exact mass (float)
                           - total net formal charge (int)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0
    total_charge = Chem.GetFormalCharge(mol)
    return rdMolDescriptors.CalcExactMolWt(mol), total_charge


def average_mass_from_smiles(smiles: str) -> float:
    """
    Computes the average molecular weight (mass with isotope abundance) 
    from a SMILES string. Matches COCONUT's average MW field.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        float: The average molecular weight, or 0.0 if RDKit cannot parse the structure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return Descriptors.MolWt(mol)


def smiles_to_fg_set(smiles: str) -> Tuple[FrozenSet[str], str]:
    """
    Matches SMARTS patterns on library structures to determine which 
    predefined functional groups are chemically present.

    Args:
        smiles (str): SMILES string of the compound.

    Returns:
        Tuple[FrozenSet[str], str]: A tuple containing:
                                    - A frozen set of matched functional group labels.
                                    - The standardized canonical SMILES string.
    """
    _mol = Chem.MolFromSmiles(smiles)
    if _mol is None:
        return frozenset(), smiles

    # Remove metals, salts, and charge variants before matching SMARTS
    mol = clean_fragment(_mol)
    if mol is None:
        return frozenset(), smiles

    matched_fgs = frozenset(
        label for label, _, _, _ in FG_IR_MAP
        if _COMPILED_SMARTS.get(label) and mol.HasSubstructMatch(_COMPILED_SMARTS[label])
    )
    
    return matched_fgs, Chem.MolToSmiles(mol)

