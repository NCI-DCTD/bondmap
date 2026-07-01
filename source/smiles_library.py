"""
SMILES Library Processing Module

This module provides tools to load, filter, process, and save chemical 
libraries derived from SMILES strings (e.g., COCONUT database). It includes 
parallelized processing using Joblib to efficiently calculate exact masses, 
adducts, and functional group sets for large datasets.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

# Local imports
from smiles_processing import smiles_to_fg_set, exact_mass_from_smiles

# Configure logging for long-running processes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ── MASS ADDUCT CONSTANTS ─────────────────────────────────────────────────────
ADDUCTS = {
    "exact"       : 0.0,      
    "protonated"  : 1.007276, # Proton mass
    "sodiated"    : 22.989218, # Sodium adduct mass
    "deprotonated": -1.007276, # Proton mass
}


# ── 1. EXACT MASS & PROCESSING TOOLS ──────────────────────────────────────────

def load_smiles_library(smiles_library_path: str | Path) -> pd.DataFrame:
    """
    Loads and normalizes a COCONUT database export.

    Expects a CSV/TSV with at minimum: identifier, smiles, molecular_weight.
    COCONUT CSV download: https://coconut.naturalproducts.net/download

    Args:
        smiles_library_path (str | Path): Path to the database file (.csv, .tsv, or .parquet).

    Returns:
        pd.DataFrame: The fully processed and loaded chemical library.
        
    Raises:
        ValueError: If the file format is unsupported.
    """
    path = Path(smiles_library_path)

    match path.suffix.lower():
        case ".csv" | ".tsv":
            sep = "\t" if path.suffix == ".tsv" else ","
            
            # Remove empty columns by filtering out 'Unnamed'
            df_smiles = pd.read_csv(
                path, 
                sep=sep, 
                low_memory=False, 
                usecols=lambda column: not column.startswith('Unnamed:')
            )
            
            # Normalize column names
            df_smiles.columns = [c.strip().lower().replace(" ", "_") for c in df_smiles.columns]

            # Ensure expected taxonomy and classification columns exist
            expected_cols = [
                "kingdom", "superclass", "class", "subclass", "organisms", 
                "np_classifier_pathway", "np_classifier_superclass", 
                "np_classifier_class", "np_classifier_is_glycoside"
            ]
            for col in expected_cols:
                if col not in df_smiles.columns:
                    df_smiles[col] = ""

            logging.info(f"Loaded {len(df_smiles):,} compounds from file.")
            
            # Process library
            df = build_library_joblib(df_smiles)
            
            # Save the processed library to parquet
            save_path = path.with_suffix(".parquet")
            logging.info(f"Saving library to {save_path}")
            save_library(df, save_path)

        case ".parquet":
            logging.info(f"Loading saved library from {path}")
            df = load_library(path)

        case _:
            raise ValueError(f"File format '{path.suffix}' not supported.")

    return df


def process_row(rec: dict) -> Optional[Dict[str, Any]]:
    """
    Processes a single compound record. 
    Must remain a top-level function to allow pickling by ProcessPoolExecutor.

    Args:
        rec (dict): A dictionary representing a single row from the dataset.

    Returns:
        Optional[Dict[str, Any]]: A parsed dictionary of compound properties, 
                                  or None if the SMILES string is invalid.
    """
    smiles = rec.get("smiles", rec.get("canonical_smiles", ""))
    if not smiles:
        return None

    try:
        fg_set, can_smiles = smiles_to_fg_set(str(smiles))
        if not fg_set:            
            return None

        ex_mw, charge = exact_mass_from_smiles(can_smiles)

        coco_id = rec.get("identifier", rec.get("id", ""))

        return {
            "smiles":                       can_smiles,
            "name":                         rec.get("name", ""),
            "id":                           coco_id,
            # stereoisomers are grouped by coconut id prefix
            "parent":                       coco_id.split('.')[0],
            "molecular_weight":             rec.get("molecular_weight", np.nan),
            "kingdom":                      rec.get("kingdom", ""),
            "superclass":                   rec.get("superclass", ""),
            "class":                        rec.get("class", ""),
            "subclass":                     rec.get("subclass", ""),
            "organisms":                    rec.get("organisms", ""),
            "fg_set":                       fg_set,
            "monoisotopic":                 ex_mw,
            # field names match adduct keys
            # [M+]; set to zero if structure is neutrally charged
            "exact":                        ex_mw / abs(charge) if charge > 0 else 0,
            # [M+H]
            "protonated":                   ex_mw + ADDUCTS.get('protonated', 0.0) if charge == 0 else ex_mw,
            # [M+Na]
            "sodiated":                     ex_mw + ADDUCTS.get('sodiated', 0.0) if charge == 0 else ex_mw,
            # [M-H]
            "deprotonated":                 ex_mw + ADDUCTS.get('deprotonated', 0.0) if charge == 0 else ex_mw,
            "source":                       "smiles_only",
            "np_classifier_pathway":        rec.get("np_classifier_pathway", ""),
            "np_classifier_superclass":     rec.get("np_classifier_superclass", ""),
            "np_classifier_class":          rec.get("np_classifier_class", ""),
            "np_classifier_is_glycoside":   rec.get("np_classifier_is_glycoside", ""),
        }
    except Exception as e:
        # Avoid spamming logs for bad smiles, but keep track
        logging.debug(f"Error processing SMILES '{smiles}': {e}")
        return None


def process_chunk(records: List[dict]) -> List[dict]:
    """
    Processes a chunk of records in a single worker call.

    Args:
        records (List[dict]): A list of compound dictionaries.

    Returns:
        List[dict]: A list of successfully processed compound dictionaries.
    """
    results = []
    for rec in records:
        result = process_row(rec)
        if result is not None:
            results.append(result)
    return results


def build_library_joblib(
    df: pd.DataFrame,
    n_workers: int = 7,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Distributes the processing of a SMILES library across multiple CPU cores.

    Args:
        df (pd.DataFrame): Raw dataframe to process.
        n_workers (int): Number of parallel jobs. Defaults to 7.
        chunk_size (int): Number of records per chunk. Defaults to 5000.

    Returns:
        pd.DataFrame: A fully processed dataframe.
    """
    records = df.to_dict("records")
    chunks = [
        records[i:i + chunk_size]
        for i in range(0, len(records), chunk_size)
    ]

    logging.info(f"Processing {len(records):,} compounds in {len(chunks)} chunks on {n_workers} workers...")
    
    results = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(process_chunk)(chunk) for chunk in chunks
    )

    rows = [row for chunk_result in results for row in chunk_result]
    logging.info(f"Done: {len(rows):,} valid compounds successfully processed.")
    
    return pd.DataFrame(rows)


def build_smiles_library(
    smiles_records: List[dict] | pd.DataFrame,
    mw_range: Optional[Tuple[float, float]] = None,
    taxonomy_filters: Optional[Dict[str, List[str]]] = None,
    organism_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds a SMILES-only library sequentially, allowing optional filtering 
    by molecular weight and taxonomy.

    Args:
        smiles_records (List[dict] | pd.DataFrame): Raw input data.
        mw_range (Optional[Tuple[float, float]]): (min_mw, max_mw) in Da.
        taxonomy_filters (Optional[Dict[str, List[str]]]): Dictionary of column -> allowed values.
        organism_filter (Optional[List[str]]): List of organism name substrings to match.

    Returns:
        pd.DataFrame: The filtered and processed library. Includes a 'filters_applied' attribute.
    """
    if isinstance(smiles_records, list):
        df = pd.DataFrame(smiles_records)
        expected_cols = [
            "name", "id", "molecular_weight", "kingdom", "superclass", 
            "class", "subclass", "organisms", "np_classifier_pathway", 
            "np_classifier_superclass", "np_classifier_class", 
            "np_classifier_is_glycoside"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = "" if col != "molecular_weight" else np.nan
    else:
        df = smiles_records.copy()

    filters_applied = {}

    # Apply Molecular Weight Filter
    if mw_range is not None:
        lo, hi = mw_range
        before = len(df)
        df = df[df["molecular_weight"].between(lo, hi, inclusive="both")]
        filters_applied["mw_range"] = f"{lo}–{hi} Da ({before - len(df):,} removed)"
        logging.info(f"MW filter {lo}–{hi} Da: {len(df):,} compounds remaining.")

    # Apply Taxonomy Filters
    if taxonomy_filters:
        for col, values in taxonomy_filters.items():
            if col not in df.columns:
                logging.warning(f"Taxonomy column '{col}' not found — skipping.")
                continue
            
            pattern = "|".join(values)
            mask = df[col].fillna("").str.contains(pattern, case=False, regex=True)
            df = df[mask]
            filters_applied[col] = values
            logging.info(f"Taxonomy filter '{col}' {values}: {len(df):,} compounds remaining.")

    # Apply Organism Filter
    if organism_filter and "organisms" in df.columns:
        pattern = "|".join(organism_filter)
        mask = df["organisms"].fillna("").str.contains(pattern, case=False, regex=True)
        df = df[mask]
        filters_applied["organisms"] = organism_filter
        logging.info(f"Organism filter {organism_filter}: {len(df):,} compounds remaining.")

    logging.info(f"Computing SMARTS FG sets for {len(df):,} compounds sequentially...")
    
    rows, skipped = [], 0
    for _, rec in df.iterrows():
        result = process_row(rec)
        if result is None:
            skipped += 1
        else:
            rows.append(result)

    if skipped:
        logging.warning(f"Skipped {skipped} compounds with invalid SMILES.")

    library_df = pd.DataFrame(rows)
    library_df.attrs["filters_applied"] = filters_applied
    
    logging.info(f"SMILES library built: {len(library_df):,} compounds.")
    return library_df


# ── 2. SAVE AND LOAD FUNCTIONS ────────────────────────────────────────────────

def save_library(df: pd.DataFrame, path: str | Path) -> None:
    """
    Serializes functional group sets and saves the DataFrame to a Parquet file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str | Path): The destination filepath.
    """
    df = df.copy()
    # Parquet cannot handle frozensets directly; convert them to pipe-separated strings
    df["fg_set"] = df["fg_set"].map(lambda x: "|".join(sorted(x)) if x else "")
    df.to_parquet(path, index=False)


def load_library(path: str | Path) -> pd.DataFrame:
    """
    Loads a library from a Parquet file and deserializes the functional groups.

    Args:
        path (str | Path): Path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded library with restored frozensets.
    """
    df = pd.read_parquet(path)
    # Restore pipe-separated strings back to Python frozensets
    df["fg_set"] = df["fg_set"].map(
        lambda x: frozenset(x.split("|")) if pd.notna(x) and x else frozenset()
    )
    return df

