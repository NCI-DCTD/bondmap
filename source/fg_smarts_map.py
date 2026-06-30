"""
FG_IR_MAP — Functional group to IR region mapping with references.

Naming convention:
    {molecule_class}_{atom_types}_{mode}
    mode: oop=out-of-plane bend, wag=wagging, bend=in-plane bend,
          symm=symmetric stretch. Absence of mode implies stretch.

Primary references:
    [P]  Pavia, Lampman, Kriz & Vyvyan, "Introduction to Spectroscopy",	5th ed., Cengage, 2015.
    [E]  Empirically adjusted on 500-compound Bruker FTIR dataset
         (natural products, this project). Deviations from literature
         noted inline.

Format:
    (identifier, SMARTS, (low_cm1, high_cm1), "notes [ref]")

Notes: https://archive.org/details/IntroductionToSpectroscopy5thEdition
       Simplified Correlation Chart [P p.29]
       https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
       SMARTS verification - https://smarts.plus/
"""

FG_IR_MAP = [

    # ── O-H / N-H STRETCH REGION (2500–3600 cm⁻¹) ───────────────────────────
    # Broad, often overlapping bands. Ranges widened empirically [E]
    # to capture hydrogen-bonded and free forms in natural products.

    ("carboxylic_acid_OH", "[CX3](=O)[OX2H1]",		                        (2500, 3300),		     "Broad O-H stretch; very broad due to H-bonding. Lit: 2400–3400 [P p.62]"),		
    ("alcohol_OH",		   "[OX2H][#6]",		                            (3200, 3400),		     "O-H stretch; free OH ~3620–3640, H-bonded 3200–3550. Lit: 3200-3400 [P p.47]. "),		
    ("primary_amine",	   "[NX3H2][#6]",		                            (3300, 3500),		     "Two N-H stretches (sym + asym); doublet characteristic. Lit: 3300–3500 [P p.74]."),		
    ("secondary_amine_NH", "[NX3H1]([#6])[#6;!$(C=O)]",                     (3310, 3350),		     "Single N-H stretch; weaker than primary. Lit: 3300–3500 [P p.74]. Excludes amide N-H (handled separately)."),		
    ("secondary_amide_NH", "[NX3H1][CX3](=O)",		                        (3150, 3330),		     "N-H stretch; shifted lower than amine due to conjugation. Lit: amide II band context [P p.74]."),		
    ("primary_amide_NH",   "[CX3](=O)[NX3H2]",		                        (3300, 3500),		     "Two N-H stretches (Amide A and B bands). Lit: 3300–3500 free [P p.74]."),		
    ("aromatic_amine_NH",  "[NX3H][cX3]",		                            (3380, 3480),		     "Ar-NH stretch; higher than aliphatic secondary amine. Lit: 3400 [P p.76]"),		
    ("pyrrole_NH",		   "[nH]",		                                    (3380, 3480),		     "Pyrrole/indole N-H stretch; sharp, characteristic. Lit: 3400 [P p.76]"),		
    
    # ── C-H STRETCH REGION (2700–3150 cm⁻¹) ─────────────────────────────────

    ("aldehyde_CH",		   "[CX3H1](=O)",		                            (2700, 2850),		     "Aldehyde C-H stretch; two weak bands (Fermi resonance). Lit: 2700–2900 [P p.56]. " 
                                                                                                     "Upper bound set to 2850 [E] to reduce overlap with alkyl C-H."),		
    ("alkyne_CH",		   "[CX2H]#[CX2]",		                            (3250, 3350),		     "Terminal alkyne ≡C-H stretch; sharp, strong. Should be a matching peak at terminal_alkyne_CC.Lit: 3300 [P p.35]."),		
    ("alkene_CH",		   "[CX3H]=[CX3]",		                            (3000, 3100),		     "Alkene =C-H stretch; above 3000 cm⁻¹. Lit: 3000–3100 [P p.33]. Upper bound extended to 3150 [E] to capture aromatic overlap."),		
    ("aromatic_CH",		   "c[cH]",		                                    (3050, 3150),		     "Aromatic C-H stretch; multiple weak bands above 3000 cm⁻¹. Lit: 3050–3150 [P p.43]. Upper bound extended to 3150 [E]."),		
    ("alkyl_CH_stretch",   "[CX4;H1,H2,H3]",	                            (2850, 3000),		     "Aliphatic C-H stretch; CH₃ asym ~2962, sym ~2872; CH₂ asym ~2926, sym ~2853. Lit: 2850–3000 [P p.31]."),		
    ("thiol_SH",		   "[#16X2H]",		                                (2500, 2600),		     "Stretch, one weak band, occurs near 2550Lit: 2550 [P p. 81]."),		
    
    # ── TRIPLE BOND REGION (2100–2260 cm⁻¹) ─────────────────────────────────
    # Often weak or absent for internal/symmetric alkynes [S p.91]

    ("nitrile",		       "[NX1]#[CX2]",		                            (2200, 2260),		     "C≡N stretch; strong and sharp. Lit: 2210–2260 [P p.77]."),		
    ("terminal_alkyne_CC", "[CX2H]#[CX2]",		                            (2100, 2250),		     "Terminal C≡C stretch; medium intensity. Should be a matching peak at alkyne_CH.Lit: 2100–2250 [P p.35]."),		
    ("internal_alkyne_CC", "[CX2]#[CX2]",		                            (2100, 2260),		     "Internal C≡C stretch; often weak or IR-inactive if symmetric. No alkyne_CHLit: 2100–2250 [P p.35]."),		
    
    # ── CARBONYL REGION (1620–1870 cm⁻¹) ─────────────────────────────────────
    # Most specific patterns listed first to reduce ambiguity.
    # Ranges widened from literature [E] to account for conjugation
    # shifts in natural products (common 20–30 cm⁻¹ red shift).

    ("anhydride_CO",	   "[CX3](=O)O[CX3](=O)",		                    (1740, 1830),		     "Anhydride high-freq C=O stretch (asym); two bands characteristic. Lit: 1740–1830 [P p.73]."),		
    ("acid_chloride",	   "[CX3](=O)[Cl]",		                            (1750, 1820),		     "Acyl chloride C=O stretch; higher than ester due to inductive effect. Should have C_Cl. Lit: 1760–1810 [P p.72]."),		
    ("ester_CO",		   "[CX3](=O)[OX2][#6]",		                    (1700, 1800),		     "Ester C=O stretch. Lit: 1715–1760 [P p.64]. Upper bound extended to 1800 [E] for acetate esters on " 
                                                                                                     "aromatic rings (~1770) and strained lactones; lower bound extended to 1700 [E] for conjugated esters."),		
    ("aldehyde_CO",		   "[CX3H1](=O)[#6]",		                        (1690, 1750),		     "Aldehyde C=O stretch. Lit: 1720–1740 [P p.56]. Lower bound extended to 1690 [E] for conjugated aldehydes."),		
    ("ketone_CO",		   "[#6][CX3](=O)[#6]",		                        (1670, 1750),		     "Ketone C=O stretch. Lit: 1680–1725 [P p.58].Range widened [E] for α,β-unsaturated ketones (~1675) and aryl ketones (~1680–1700)."),		
    ("carboxylic_acid_CO", "[CX3](=O)[OX2H1]",		                        (1680, 1730),		     "Carboxylic acid C=O stretch. Lit: 1700–1725 [P p.62]. Lower bound extended to 1680 [E] for conjugated acids."),		
    ("primary_amide_CO",   "[CX3](=O)[NX3H2]",		                        (1630, 1700),		     "Primary amide Amide I band (C=O stretch). Should show primary_amide_NH.Lit: 1640–1700 [P p.70]."),		
    ("secondary_amide_CO", "[CX3](=O)[NX3H1]",		                        (1630, 1700),		     "Secondary amide Amide I band (C=O stretch). Should show secondary_amide_NH.Lit: 1640–1700 [P p.70]."),		
    ("conjugated_CO",	   "[CX3](=O)[cX3]",		                        (1620, 1690),		     "Aryl/conjugated C=O stretch; red-shifted vs unconjugated. Lit: 1630–1700 [P p.75]. " 
                                                                                                     "Lower bound extended to 1620 [E] for chromone/flavone systems."),		
    
    # ── C=C / C=N REGION (1620–1680 cm⁻¹) ───────────────────────────────────

    ("aromatic_CC",		   "c1ccccc1",		                                (1475, 1600),		     "Aromatic C=C ring stretch; two bands ~1500 and ~1600. Lit: 1475–1600 [P p.43]."),		
    ("alkene_CC",		   "[CX3]=[CX3]",		                            (1620, 1680),		     "C=C stretch; intensity varies with substitution symmetry. Lit: 1600–1680 [P p.33]."),		
    ("imine_CN",		   "[CX3;!$(C=O)]=[NX2;!$(N-O)]",                   (1620, 1690),		     "C=N imine stretch. Lit: 1640–1690 [P p.77]. Excludes oximes and carbonyls."),		
    ("oxime_CN",		   "[CX3]=[NX2][OX2H]",		                        (1630, 1690),		     "C=N oxime stretch; similar position to imine. Lit: 1640–1690 [P p.77]."),		
    ("amidine_CN",		   "[NX3][CX3]=[NX2]",		                        (1620, 1700),		     "Amidine/guanidine C=N stretch; multiple bands. Lit: ."),		
    
    # ── NITROGEN (1300–1570 cm⁻¹) ─────────────────────────────────────────────

    ("nitro",		       "[$([NX3](=O)=O),$([NX3+](=O)[O-])][#6]",        (1500, 1600),		     "Nitro N=O asymmetric stretch. Lit: 1530–1600 [P p.79]."),		
    ("nitro_symm",		   "[$([NX3](=O)=O),$([NX3+](=O)[O-])][#6]",        (1300, 1390),		     "Nitro N=O symmetric stretch. Lit: 1300–1390 [P p.79]."),		
    
    # ── C-O / C-N SINGLE BOND REGION (1000–1310 cm⁻¹) ───────────────────────

    ("ester_CO_single",	   "[CX3](=O)[OX2][#6]",		                    (1100, 1300),		     "Ester C-O single bond stretch (C-O-C); strong band. Lit: 1150–1250 [P p.65]. Wide range [E] covers both acyl C-O (~1250) and alkyl C-O (~1050)."),		
    ("alcohol_CO",		   "[OX2H][CX4]",		                            (1000, 1260),		     "Alcohol C-O stretch; position varies with substitution. Primary ~1050, secondary ~1100, tertiary ~1150. Lit: 1000–1260 [P p.47]."),		
    ("ether_CO",		   "[OX2]([#6])[#6]",		                        (1000, 1150),		     "Ether C-O-C asymmetric stretch; strong. Lit: 1000–1300 [P p.50]."),		
    ("aryl_ether_CO",	   "[OX2]([cX3])[#6]",		                        (1200, 1310),		     "Aryl ether C-O stretch; shifted higher than aliphatic ether. Lit: 1220–1310 [P p.52]. Added [E] after unmatched peak analysis at 1250–1300."),		
    ("amine_CN",		   "[NX3][CX4]",		                            (1020, 1300),		     "Amine C-N single bond stretch. Primary ~1020–1090, secondary ~1090–1150, tertiary ~1150–1220. Lit: 1000–1350 [P p.74]."),		
    
    # ── C-H BENDING REGION (700–1480 cm⁻¹) ──────────────────────────────────

    ("aromatic_CH_oop",	   "c[cH]",		                                    (700, 900),		         "Aromatic C-H out-of-plane bend; strong, position indicates substitution pattern. Monosub ~770, para ~820. Lit: 690–900 [P p.43]."),		
    ("alkene_CH_wag",	   "[CX3H]=[CX3]",		                            (650, 1000),		     "Alkene =C-H out-of-plane wag; strong. Terminal =CH₂ ~910, 990. Lit: 650–1000 [P p.33]."),		
    ("CH2_CH3_bend",	   "[CX4;H2,H3]",		                            (1350, 1480),		     "CH₂ scissor ~1465, CH₃ asym bend ~1450, CH₃ sym (umbrella) ~1375. 1375–1465 [P p.32]. "),		
    
    # ── SULFUR (600–1350 cm⁻¹) ───────────────────────────────────────────────

    ("sulfoxide_SO",	   "[#6][SX3](=O)[#6]",		                        (1025, 1075),		     "S=O stretch; strong. Lit: 1050 [P p.81]."),		
    ("sulfone_SO",		   "[#6][SX4](=O)(=O)[#6]",		                   [(1125, 1175), 
                                                                            (1275, 1325)],		     "SO₂ symmetric stretch; asymmetric ~1300. Lit: 1150, 1300 [P p.82]."),		
    ("sulfonate_SO",	   "[$([SX4](=O)(=O)[OX2]),$([SX4](=O)(=O)[O-])]", [(1325, 1375), 
                                                                            (1150, 1175), 
                                                                            (750, 1000)],           "Sulfonate S=O stretch (asymm, symm). Single S-O stretch. Lit: 750-1000, 1175, 1350 [P p.82]. "),		
    ("sulfonamide_SN",	   "[#6][SX4](=O)(=O)[NX3]",		               [(1300, 1350),
                                                                            (1115, 1165),
                                                                            (1525, 1575)],          "Sulfonate S=O stretch (asymm, symm). Single N-H stretch. Lit: 1550, 1140, 1325 [P p.82]. "),		
    
    # ── HALOGENS (500–1400 cm⁻¹) ─────────────────────────────────────────────

    ("C_F",		           "[CX4][F]",		                                (1000, 1400),		     "C-F stretch; very strong, wide range. Lit: 1000–1400 [P p.85]."),		
    ("C_Cl",		       "[CX4][Cl]",		                                (600, 800),		         "C-Cl stretch; strong. Lit: 540–785 [P p.85]."),		
    ("C_Br",		       "[CX4][Br]",		                                (500, 650),		         "C-Br stretch; strong. Lit: 510–650 [P p.85]."),		]

'''
The labels are used in multiple places
'''
FG_LABELS = [label for label, *_ in FG_IR_MAP]
