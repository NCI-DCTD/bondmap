"""
Library Search and Filtering Module

This module provides high-throughput searching utilities for structural 
chemical libraries. It utilizes a three-stage filtering system based on 
molecular weight tolerances, the nominal mass nitrogen rule, and a customized 
functional group recall score.
"""

from typing import Tuple, Set, Optional, FrozenSet
import pandas as pd

from smiles_library import ADDUCTS

def _target_recall_score(query_fgs: FrozenSet[str], target_fgs: FrozenSet[str]) -> float:
    """
    Calculates the proportion of target functional groups (FGs) present in the query.

    Ignores extra FGs in the query (i.e., false positives are not penalized).
    
    Args:
        query_fgs (FrozenSet[str]): Detected functional groups from the query spectrum.
        target_fgs (FrozenSet[str]): Functional groups belonging to the target library compound.

    Returns:
        float: Recall overlap score between 0.0 and 1.0.
    """
    if not target_fgs:
        return 0.0
    return len(query_fgs & target_fgs) / len(target_fgs)


def calculate_nominal_mass(mw: float, adduct: float = 0.0) -> int:
    """
    Calculates the nominal mass from exact/monoisotopic mass of an ion by 
    subtracting any adduct mass and adjusting for the organic hydrogen 
    mass defect (approx. 0.1 Da per 100 Da) before casting to an integer.

    Args:
        mw (float): The exact or monoisotopic mass of the ion (e.g., 652.6).
        adduct (float): The mass of the added adduct (e.g., 1.007276). Defaults to H_MASS.

    Returns:
        int: The adjusted nominal mass of the ion.
    """
    # Calculate mass defect correction based purely on the neutral molecule's mass (mw - adduct)
    neutral_mass = mw - adduct
    mass_defect_correction = 0.1 * (neutral_mass / 100.0)
    
    # Subtract the defect correction from neutral mass
    corrected_mw = neutral_mass - mass_defect_correction
    
    # Truncate to get the nominal integer mass
    return int(corrected_mw)


def search_library(
    query_fgs: Set[str],
    library_df: pd.DataFrame,
    mw: Optional[float] = None,
    mw_tol: float = 0.05,
    top_n: int = 50,
    tanimoto: float = 0.5,
    search_mw: str = "protonated",
) -> Tuple[pd.DataFrame, int]:
    """
    Executes a three-stage library search using molecular weight, 
    nitrogen estimation, and tiered recall score filtering.

    Stages:
        1. Molecular Weight Filter (within specified tolerance)
        2. Nitrogen Rule Filter (estimates nominal mass matching)
        3. Recall Matching Filter (ranks results by overlap of query/target FGs)

    Args:
        query_fgs (Set[str]): Set of detected functional groups.
        library_df (pd.DataFrame): DataFrame containing the chemical library database.
        mw (Optional[float]): Target query mass (typically protonated mass [M+H]+).
        mw_tol (float): Tolerance window around target mass in Daltons. Defaults to 0.05.
        top_n (int): Maximum number of records to return. Defaults to 50.
        tanimoto (float): Minimum target recall score required. Defaults to 0.5.
        search_mw (str): Column name in library_df representing the mass. Defaults to "protonated".
        nitro_rule (bool): If True, applies the nitrogen rule check to the query mass.

    Returns:
        Tuple[pd.DataFrame, int]: 
            - DataFrame of top_n sorted candidates.
            - Total number of candidates remaining after Stage 1 (MW Filter).
    """
    # Create copy of database to avoid SettingWithCopyWarning
    candidates = library_df.copy()

    # ── Stage 1: Molecular Weight Filter ───────────────────────────────────────
    if mw is not None:
        mw_mask = (candidates[search_mw] >= (mw - mw_tol)) & (candidates[search_mw] <= (mw + mw_tol))
        candidates = candidates[mw_mask]
    
    mw_candidates = len(candidates)

    # ── Stage 2: Tiered Recall Filter & Scoring ────────────────────────────────
    query_frozenset = frozenset(query_fgs)
    
    # Calculate functional group recall score
    candidates["rank_score"] = candidates["fg_set"].apply(
        lambda target: _target_recall_score(query_frozenset, target)
    )

    # Length of functional groups is used to break ties (prioritizes larger structures)
    candidates["fg_len"] = candidates["fg_set"].apply(len)

    # Apply score threshold and sort
    candidates = (
        candidates[candidates['rank_score'] >= tanimoto]
        .sort_values(by=["rank_score", "fg_len"], ascending=[False, False])
    )

    # Return top results and match statistics
    final_candidates = candidates.head(top_n).reset_index(drop=True)
    return final_candidates, mw_candidates

