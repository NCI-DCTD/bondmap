"""
Report Generation and Visualization Module

This module provides utilities to visualize infrared spectra, render molecular 
structures via RDKit, and generate comprehensive HTML reports for spectral 
dereplication and structural candidate matching.
"""

import base64
import colorsys
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Any, FrozenSet, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs

# Local imports
from fg_smarts_map import FG_IR_MAP, FG_LABELS
from smiles_processing import smiles_to_fg_set


def fig_to_base64(fig: Figure) -> str:
    """
    Converts a matplotlib Figure to a base64 encoded PNG string.

    Args:
        fig (Figure): The matplotlib figure to encode.

    Returns:
        str: Base64 encoded string of the image.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def mol_to_base64(smiles: str, size: Tuple[int, int] = (200, 200)) -> Optional[str]:
    """
    Generates a 2D drawing of a molecule from a SMILES string and encodes it to base64.

    Args:
        smiles (str): SMILES string of the molecule.
        size (Tuple[int, int]): Dimensions of the generated image.

    Returns:
        Optional[str]: Base64 encoded PNG string, or None if RDKit fails to parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    buf = BytesIO()
    Draw.MolToImage(mol, size=size).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_colors_hsv(n: int) -> List[Tuple[float, float, float]]:
    """
    Generates an evenly spaced array of distinct colors using the HSV color space.
    
    Args:
        n (int): Number of colors to generate.

    Returns:
        List[Tuple[float, float, float]]: List of RGB tuples.
    """
    return [colorsys.hsv_to_rgb(i / n, 0.7, 0.9) for i in range(n)]

# Generate a consistent color palette mapping to all functional group labels
colors = make_colors_hsv(len(FG_LABELS))


def plot_spectrum(
    wn: np.ndarray, 
    absorbance: np.ndarray, 
    title: str = "IR Spectrum", 
    smiles: Optional[str] = None, 
    show_regions: Optional[FrozenSet[str]] = None, 
    transmittance: bool = False, 
    show_legend: bool = True
) -> Figure:
    """
    Plots an infrared spectrum with highlighted functional group regions.

    Args:
        wn (np.ndarray): Array of wavenumbers.
        absorbance (np.ndarray): Array of absorbance (or transmittance) values.
        title (str): Title of the plot.
        smiles (Optional[str]): SMILES string of the compound (unused directly in plotting).
        show_regions (Optional[FrozenSet[str]]): Specific FG regions to highlight on the plot.
        transmittance (bool): If True, inverts the Y-axis and labels it Transmittance.
        show_legend (bool): Whether to display the plot legend.

    Returns:
        Figure: The generated matplotlib figure.
    """
    fig, ax_ir = plt.subplots(figsize=(12, 4))

    # --- spectrum regions ---
    if show_regions:
        regions_to_plot = [
            (_regions, color, fg) 
            for (fg, _, _regions, label), color in zip(FG_IR_MAP, colors) 
            if fg in show_regions
        ]
        
        for region, color, fg in regions_to_plot:
            if isinstance(region[0], (int, float)):
                region = [region]  # single tuple (lo, hi) → wrap in list
            
            for lo, hi in region:
                lo_c = max(lo, wn.min())
                hi_c = min(hi, wn.max())
                if lo_c < hi_c:
                    ax_ir.axvspan(lo_c, hi_c, alpha=0.25, color=color, label=fg)

    # --- plot main line ---
    ax_ir.plot(wn, absorbance, color="steelblue", linewidth=0.9)
    ax_ir.set_xlim(wn.max(), wn.min())
    ax_ir.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)

    if transmittance:
        ax_ir.invert_yaxis()
        ax_ir.set_ylabel("Transmittance", fontsize=12)
    else:
        ax_ir.set_ylabel("Absorbance", fontsize=12)
    
    if show_legend:
        ax_ir.legend(loc="upper left", fontsize=9, framealpha=0.7)
    
    ax_ir.grid(True, linestyle="--", alpha=0.4)
    return fig


def build_report_html(
    spectrum_fig: Figure,
    query_fgs: frozenset,
    candidates: pd.DataFrame,
    query_name: str = "Unknown",
    top_n: int = 10,
    output_path: str = "report.html",
    open_browser: bool = True,
    search_stats: Optional[Dict[str, int]] = None,
    search_param: Optional[Dict[str, Any]] = None,
    training_data: Optional[List[Any]] = None,
    FG_CONFIDENCE: Optional[Dict[str, str]] = None,
) -> str:
    """
    Builds an HTML dereplication report and writes it to disk.

    Args:
        spectrum_fig (Figure): Matplotlib figure of the target IR spectrum.
        query_fgs (frozenset): Set of detected functional group labels.
        candidates (pd.DataFrame): DataFrame of candidate structures. Expects columns: 
                                   smiles, name/id, rank_score, molecular_weight.
        query_name (str): Title for the report.
        top_n (int): Maximum number of candidate structures to display.
        output_path (str): Filepath where the HTML file should be saved.
        open_browser (bool): Automatically open the generated HTML in the default browser.
        search_stats (Optional[Dict]): Library reduction statistics.
        search_param (Optional[Dict]): Search parameter thresholds.
        training_data (Optional[List]): List of training dataset objects for Tanimoto comparison.
        FG_CONFIDENCE (Optional[Dict]): Mapping of FG labels to their confidence tier.

    Returns:
        str: Absolute path to the written HTML report.
    """
    # Prevent NoneType AttributeError 
    FG_CONFIDENCE = FG_CONFIDENCE or {}

    # ── 1. Spectrum Rendering ──────────────────────────────────────────────────
    spectrum_b64 = fig_to_base64(spectrum_fig)

    # ── 2. Functional Group Badges ─────────────────────────────────────────────
    def fg_tier(fg: str) -> str:
        return FG_CONFIDENCE.get(fg, "none")

    fg_items = "".join(
        f'<li><span class="badge {fg_tier(fg)}">{fg_tier(fg).upper()}</span>{fg}</li>'
        for fg in sorted(query_fgs)
    )
    
    # ── 3. Search Parameters & Tanimoto Structural Evaluation ──────────────────
    search_html = ""
    tanimoto_html = ""
    training_html = ""

    if search_param:
        search_html = f"""
        <p class="stats">
            Search Parameters: 
            <span>MW: {search_param.get('mw', 0):,}</span> →
            <span>Target MW: {search_param.get('search_mw', '')}</span> →
            <span>FG tanimoto: {search_param.get('tanimoto', 0):,}</span>
        </p>
        """
        
        test_smiles = search_param.get("test_smi")
        
        if test_smiles:
            test_mol = Chem.MolFromSmiles(test_smiles)
            if test_mol:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
                hits = []
                
                for i, row in candidates.iterrows():
                    mol2 = Chem.MolFromSmiles(row["smiles"])
                    if mol2 is None:
                        continue
                    
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                    
                    if similarity >= 0.8:
                        hits.append({
                            "rank": i + 1,
                            "id": row.get("id", "—"),
                            "similarity": similarity,
                            "rank_score": row["rank_score"],
                        })
                
                test_mol_b64 = mol_to_base64(test_smiles)
                test_img = f'<img src="data:image/png;base64,{test_mol_b64}" class="test-mol"/>' if test_mol_b64 else ""
                
                if hits:
                    rows = "".join(f"""
                        <tr>
                            <td>{h['rank']}</td>
                            <td>{h['id']}</td>
                            <td>{h['similarity']:.3f}</td>
                            <td>{h['rank_score']:.3f}</td>
                        </tr>""" for h in hits)
                    
                    tanimoto_html = f"""
                    <h2>Structural Evaluation (Tanimoto ≥ 0.8)</h2>
                    <div class="tanimoto-block">
                        {test_img}
                        <table class="tanimoto-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>ID</th>
                                    <th>Tanimoto</th>
                                    <th>IR Score</th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                    """
                else:
                    tanimoto_html = f"""
                    <h2>Structural Evaluation (Tanimoto ≥ 0.8)</h2>
                    <div class="tanimoto-block">
                        {test_img}
                    </div>
                    <p class="no-hits">No candidates with Tanimoto ≥ 0.8 found in results.</p>
                    """

            # ── 4. Training Data Match ─────────────────────────────────────────────
            if training_data and test_mol:
                training_hits = []
                for i, d in enumerate(training_data):
                    mol2 = Chem.MolFromSmiles(d.smiles)
                    if mol2 is None:
                        continue
                        
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                    
                    if similarity >= 0.8:
                        training_hits.append((i, similarity, d))
                
                training_hits.sort(key=lambda x: x[1], reverse=True)
                
                if training_hits:
                    cards = ""
                    for i, similarity, d in training_hits:
                        struct_fgs = smiles_to_fg_set(d.smiles)[0]
                        train_fig = plot_spectrum(
                            d.wn, d.absorbance,
                            title=d.sample_id, smiles=d.smiles,
                            show_regions=struct_fgs, 
                        )
                        train_spec_b64 = fig_to_base64(train_fig)
                        mol_b64 = mol_to_base64(d.smiles)
                        
                        img_tag = f'<img class="train-mol" src="data:image/png;base64,{mol_b64}"/>' if mol_b64 else ""
                        exact = "✓ Exact match" if similarity == 1.0 else ""
                        
                        cards += f"""
                        <div class="train-card">
                            <div class="train-card-header">
                                <span class="train-id">{d.sample_id}</span>
                                <span class="train-sim">Tanimoto: {similarity:.3f}</span>
                                <span class="train-exact">{exact}</span>
                            </div>
                            <div class="train-card-body">
                                {img_tag}
                                <img class="train-spectrum" src="data:image/png;base64,{train_spec_b64}"/>
                            </div>
                        </div>
                        """
                        plt.close(train_fig)  # Prevent matplotlib memory leak
                        
                    training_html = f"""
                    <h2>Training Data Match</h2>
                    {cards}
                    """
                else:
                    training_html = """
                    <h2>Training Data Match</h2>
                    <p class="no-hits">No training compounds with Tanimoto ≥ 0.8.</p>
                    """
        
    # ── 5. Search Statistics ───────────────────────────────────────────────────
    stats_html = ""
    if search_stats:
        stats_html = f"""
        <p class="stats">
            Library reduction: 
            <span>MW filter: {search_stats.get('mw_count', 0):,}</span> →
            <span>FG match: {search_stats.get('fg_count', 0):,}</span>
        </p>
        """

    # ── 6. Candidate Cards ─────────────────────────────────────────────────────
    candidate_cards = ""
    for i, row in candidates.head(top_n).iterrows():
        mol_b64 = mol_to_base64(row["smiles"])
        img_tag = f'<img src="data:image/png;base64,{mol_b64}"/>' if mol_b64 else '<p class="no-mol">No structure</p>'
        
        name = row.get("name", row.get("id", "—"))
        cid = row.get("id", "—")
        struct_fgs = row.get("fg_set", [])
        
        fg_tags = "".join(
            f'<li><span class="badge {fg_tier(fg)}">{fg_tier(fg).upper()}</span>{fg}</li>'
            for fg in sorted(struct_fgs)
        )
        
        candidate_cards += f"""
        <div class="card">
            <div class="card-title">{i+1}</div>
            {img_tag}
            <div class="card-body">
                <p class="card-name">{cid}</p>
                <p class="card-name">{name}</p>
                <p class="card-score">Score: {row['rank_score']:.3f}</p>
                <p class="card-mw">MW: {row['molecular_weight']:.2f}</p>
                <p class="card-smiles" title="{row['smiles']}">{row['smiles'][:40]}…</p>
                {fg_tags}
            </div>
        </div>
        """

    # ── 7. Generate Final HTML ─────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IR Report — {query_name}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: Arial, sans-serif;
        background: #f5f5f5;
        color: #333;
        padding: 32px;
    }}
    h1 {{ font-size: 1.6em; color: #2c3e50; margin-bottom: 8px; }}
    h2 {{
        font-size: 1.1em; color: #34495e;
        border-bottom: 2px solid #ddd;
        padding-bottom: 4px;
        margin: 24px 0 12px;
    }}
    .stats {{
        font-size: 0.85em; color: #7f8c8d;
        margin-top: 6px; margin-bottom: 4px;
    }}
    .stats span {{
        background: #ecf0f1; border-radius: 4px;
        padding: 2px 8px; margin: 0 2px;
        color: #2c3e50; font-weight: bold;
    }}
    .spectrum img {{ width: 100%; border-radius: 6px; }}
    
    /* FG list */
    .fg-list {{ list-style: none; columns: 3; gap: 8px; }}
    .fg-list li {{ padding: 3px 0; font-size: 0.9em; }}
    .badge {{
        display: inline-block;
        font-size: 0.7em; font-weight: bold;
        padding: 1px 5px; border-radius: 3px;
        margin-right: 6px; vertical-align: middle;
    }}
    .complete {{ background: #2ecc71; color: white; }}
    .high     {{ background: #3498db; color: white; }}
    .medium   {{ background: #f39c12; color: white; }}
    .low      {{ background: #bdc3c7; color: #333;  }}
    .none     {{ display: none;  }}
    
    /* candidate cards */
    .candidates {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; }}
    .card {{
        background: white; border: 1px solid #ddd;
        border-radius: 8px; width: 200px;
        overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .card img {{ width: 100%; }}
    .card-title {{ padding: 8px; font-weight: bold; background: #bdc3c7; color: #333; }}
    .card-body {{ padding: 8px; }}
    .card-name  {{ font-weight: bold; font-size: 0.8em; margin-bottom: 4px; }}
    .card-score {{ color: #27ae60; font-size: 0.8em; }}
    .card-mw    {{ color: #7f8c8d; font-size: 0.75em; }}
    .card-smiles {{
        color: #aaa; font-size: 0.65em; margin-top: 4px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .no-mol {{ color: #aaa; text-align: center; padding: 16px; }}
    
    /* Test smiles vs hits scores */
    .tanimoto-block {{ display: flex; align-items: flex-start; gap: 24px; margin-top: 8px; }}
    .test-mol {{ width: 180px; border: 1px solid #ddd; border-radius: 6px; }}
    .tanimoto-table {{ border-collapse: collapse; font-size: 0.85em; flex: 1; }}
    .tanimoto-table th {{
        background: #2c3e50; color: white;
        padding: 6px 12px; text-align: left;
    }}
    .tanimoto-table td {{ padding: 5px 12px; border-bottom: 1px solid #eee; }}
    .tanimoto-table tr:hover td {{ background: #f0f7ff; }}
    .no-hits {{ color: #aaa; font-style: italic; }}
    
    .train-card {{
        background: white; border: 1px solid #ddd;
        border-radius: 8px; margin-bottom: 16px;
        overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .train-card-header {{
        background: #2c3e50; color: white;
        padding: 8px 16px; display: flex; gap: 16px; align-items: center;
    }}
    .train-id     {{ font-weight: bold; font-size: 0.9em; }}
    .train-sim    {{ font-size: 0.85em; color: #bdc3c7; }}
    .train-exact  {{ font-size: 0.85em; color: #2ecc71; font-weight: bold; }}
    .train-card-body {{ display: flex; align-items: flex-start; gap: 16px; padding: 12px; }}
    .train-mol      {{ width: 160px; border: 1px solid #eee; border-radius: 4px; }}
    .train-spectrum {{ flex: 1; border-radius: 4px; }}
</style>
</head>
<body>
    <h1>IR Dereplication Report — {query_name}</h1>
    {search_html}
    <h2>Spectrum</h2>
    <div class="spectrum">
        <img src="data:image/png;base64,{spectrum_b64}" alt="IR Spectrum"/>
    </div>
    <h2>Detected Functional Groups ({len(query_fgs)})</h2>
    <ul class="fg-list">{fg_items}</ul>
    
    {stats_html}
    
    <h2>Top {top_n} Candidates</h2>
    <div class="candidates">{candidate_cards}</div>
    
    {tanimoto_html}
    
    {training_html}
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    
    print(f"Report written to {path.resolve()}")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
        
    return str(path.resolve())

