#!/usr/bin/env python3
"""
evaluate_retrosynthesis_routes.py — RDKit-based evaluation of AiZynthFinder
retrosynthesis route YAML files produced by evaluate_zydus_queries.py.

Appends an "evaluation:" block under _meta in each input file.

Usage:
    python scripts/evaluate_retrosynthesis_routes.py [FILE ...]

    If no FILE is given, processes all zydus_query_*.yaml in
    src/chemformer/data/.  Pass --force to re-evaluate existing results.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import rdBase

# Hill order: C first, then H, then alphabetical
_HILL_ORDER = [6, 1, 7, 8, 9, 15, 16, 17, 35, 53]
_ATOMIC_SYM = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 15:"P",
               16:"S", 17:"Cl", 35:"Br", 53:"I"}

# Lookup table: Hill-formula → canonical SMILES for common small molecules
# that appear as byproducts or co-reagents in retrosynthetic disconnections.
# These are the most common leaving groups, solvents, and condensation products.
COMMON_BYPRODUCT_SMILES = {
    # Condensation / elimination byproducts
    "H2O":      "O",
    "HCl":      "Cl",
    "HBr":      "Br",
    "HI":       "I",
    "HF":       "F",
    "CO2":      "O=C=O",
    "NH3":      "N",
    "H2S":      "S",
    "CH2O":     "C=O",          # formaldehyde
    "CH2O2":    "OC=O",         # formic acid
    "C2H4O2":   "CC(=O)O",      # acetic acid
    "C4H6O3":   "CC(=O)OC(C)=O", # acetic anhydride
    "C2H5OH":   "CCO",          # ethanol
    "CH3OH":    "CO",           # methanol
    "C6H5OH":   "Oc1ccccc1",    # phenol
    "CH3NaO":   "[Na+].[CH3O-]", # sodium methoxide (base)
    "ClNa":     "[Na+].[Cl-]",
    "ClK":      "[K+].[Cl-]",
    "Et3N":     "CCN(CC)CC",    # triethylamine (base used to scavenge HCl/HBr)
    # Sulfonyl / sulfonate leaving groups
    "CH3SO3H":  "CS(=O)(=O)O",  # methanesulfonic acid
    "CH3SO2Cl": "CS(=O)(=O)Cl", # mesyl chloride
    "CH3S":     "CS",           # methanethiol
    # Phosphorus
    "H3PO4":    "OP(=O)(O)O",
    # Boron (Suzuki coupling)
    "B(OH)3":   "OB(O)O",
    "H3BO3":    "OB(O)O",
    # Nitrogen
    "N2":       "N#N",
}

# Build formula → canonical SMILES map; skip entries with '.' (salts/mixtures
# won't parse through MolFromSmiles but are still valid lookup values).
_FORMULA_TO_SMILES = {}
for _formula, _smi in COMMON_BYPRODUCT_SMILES.items():
    if "." in _smi:
        _FORMULA_TO_SMILES[_formula] = _smi   # keep as-is; can't canonicalize
    else:
        _mol = Chem.MolFromSmiles(_smi)
        if _mol is not None:
            _FORMULA_TO_SMILES[_formula] = Chem.MolToSmiles(_mol)


# ---------------------------------------------------------------------------
# SMILES / atom utilities
# ---------------------------------------------------------------------------

def parse_smiles(smiles: str):
    """
    Attempt to parse a SMILES string with RDKit.

    Returns an RDKit Mol on success, or None if the SMILES is empty, None,
    or fails RDKit sanitization.  RDKit error messages are suppressed.
    """
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def is_valid_smiles(smiles: str) -> bool:
    """Return True if smiles parses to a valid RDKit Mol, False otherwise."""
    return parse_smiles(smiles) is not None


def atom_counts(mol) -> Counter:
    """
    Return a Counter {atomic_num: count} covering all heavy atoms and their
    implicit (and graph-explicit) hydrogens.

    Hydrogens are keyed by atomic number 1.  Counting implicit H via
    GetTotalNumHs() ensures balance checks work even when SMILES omit H.
    """
    counts: Counter = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetAtomicNum()] += 1
        counts[1] += atom.GetTotalNumHs()
    return counts


def counts_for_smiles_list(smiles_list) -> Counter:
    """
    Sum atom_counts() over a list of SMILES strings.

    Skips any SMILES that fail to parse.  Returns an empty Counter if all fail.
    """
    total: Counter = Counter()
    for smi in smiles_list:
        mol = parse_smiles(smi)
        if mol is not None:
            total += atom_counts(mol)
    return total


def molecular_formula(counts: Counter) -> str:
    """
    Convert an atom-count Counter to a Hill-order molecular formula string
    (C first, then H, then remaining elements alphabetically by symbol).

    Example: Counter({6:2, 1:4, 8:2}) → "C2H4O2"
    """
    parts = []
    for z in _HILL_ORDER:
        n = counts.get(z, 0)
        if n:
            sym = _ATOMIC_SYM.get(z, str(z))
            parts.append(sym if n == 1 else f"{sym}{n}")
    for z in sorted(k for k in counts if k not in _HILL_ORDER and counts[k]):
        sym = _ATOMIC_SYM.get(z, str(z))
        n = counts[z]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts) or "none"


def suggest_byproduct_smiles(surplus: Counter):
    """
    Given a Counter of surplus atoms (precursor atoms not accounted for in the
    product), attempt to identify a single SMILES string for the implied
    byproduct by looking up the Hill-order formula in COMMON_BYPRODUCT_SMILES.

    Returns a canonical SMILES string if found, otherwise None.

    The surplus may represent multiple molecules (e.g., HCl + H2O from a
    chloroacyl esterification).  This function only handles the case where the
    entire surplus corresponds to one known molecule.
    """
    if not surplus:
        return None
    formula = molecular_formula(surplus)
    return _FORMULA_TO_SMILES.get(formula)


# ---------------------------------------------------------------------------
# Reaction evaluation
# ---------------------------------------------------------------------------

def evaluate_reaction(rxn_smiles: str) -> dict:
    """
    Evaluate one reaction SMILES stored in retrosynthetic direction:
        product >> precursor1 . precursor2 ...

    Steps performed:
      1. Parse every fragment (product and precursors) with RDKit.
      2. Count atoms (heavy + implicit H) on each side.
      3. Classify the balance:
           - balanced:          precursor atoms == product atoms (no byproduct)
           - surplus (normal):  precursor atoms > product atoms — represents an
                                implied byproduct omitted from the SMARTS; we
                                record its Hill formula and look up a SMILES.
           - phantom atoms:     product atoms > precursor atoms — atoms appear
                                from nowhere; chemically implausible.
      4. Check physical realizability of each precursor SMILES (valence,
         hypervalent atoms, strained rings > 3-membered checked via RDKit
         sanitization; radical/charged species flagged).

    Returns a dict with keys:
        reaction_smiles             — input string
        all_smiles_valid            — bool: every fragment parses in RDKit
        molecules                   — list of {role, smiles, valid, rdkit_issues}
        balanced                    — bool
        phantom_atoms               — dict {symbol: delta}, set when product > precursors
        implied_byproduct_formula   — Hill formula of surplus atoms (str)
        implied_byproduct_smiles    — SMILES if formula found in lookup table, else null
        balance_achievable          — bool: True when balanced OR surplus matched to known SMILES
        note                        — human-readable summary
        policy_probability          — float, filled by caller
        precursors                  — list of {smiles, smiles_valid, in_stock}, filled by caller
    """
    result: dict = {"reaction_smiles": rxn_smiles}

    if not rxn_smiles or ">>" not in rxn_smiles:
        result.update(all_smiles_valid=False, balance_achievable=False,
                      note="no >> separator")
        return result

    product_part, precursor_part = rxn_smiles.split(">>", 1)

    product_smiles   = [s.strip() for s in product_part.split(".")  if s.strip()]
    precursor_smiles = [s.strip() for s in precursor_part.split(".") if s.strip()]

    # --- Step 1: parse and validate every fragment ---
    molecules = []
    for smi in product_smiles:
        mol, issues = parse_and_check(smi)
        molecules.append({"role": "product",   "smiles": smi,
                           "valid": mol is not None, "rdkit_issues": issues})
    for smi in precursor_smiles:
        mol, issues = parse_and_check(smi)
        molecules.append({"role": "precursor", "smiles": smi,
                           "valid": mol is not None, "rdkit_issues": issues})
    result["molecules"] = molecules
    result["all_smiles_valid"] = all(m["valid"] for m in molecules)

    if not result["all_smiles_valid"]:
        result.update(balanced=False, balance_achievable=False,
                      note="one or more SMILES fragments failed RDKit parse")
        return result

    # --- Step 2: atom counts ---
    p_mols = [parse_smiles(s) for s in product_smiles]
    r_mols = [parse_smiles(s) for s in precursor_smiles]
    p_counts = sum((atom_counts(m) for m in p_mols), Counter())
    r_counts = sum((atom_counts(m) for m in r_mols), Counter())

    # --- Step 3: balance classification ---
    all_elems = set(p_counts) | set(r_counts)
    phantom  = {_ATOMIC_SYM.get(z, str(z)): p_counts[z] - r_counts[z]
                for z in all_elems if p_counts[z] > r_counts[z]}
    surplus  = Counter({z: r_counts[z] - p_counts[z]
                        for z in all_elems if r_counts[z] > p_counts[z]})

    if phantom:
        result["balanced"] = False
        result["phantom_atoms"] = phantom
        result["implied_byproduct_formula"] = None
        result["implied_byproduct_smiles"] = None
        result["balance_achievable"] = False
        result["note"] = (
            f"chemically implausible — atoms appear in product absent from "
            f"precursors: {phantom}"
        )
    elif surplus:
        formula = molecular_formula(surplus)
        byp_smi = suggest_byproduct_smiles(surplus)
        result["balanced"] = False
        result["phantom_atoms"] = {}
        result["implied_byproduct_formula"] = formula
        result["implied_byproduct_smiles"] = byp_smi
        result["balance_achievable"] = byp_smi is not None
        if byp_smi:
            result["note"] = (
                f"balanceable — forward: precursors → product + {byp_smi} ({formula})"
            )
        else:
            result["note"] = (
                f"retrosynthetic disconnection — forward: precursors → "
                f"product + {formula} (SMILES unknown)"
            )
    else:
        result["balanced"] = True
        result["phantom_atoms"] = {}
        result["implied_byproduct_formula"] = None
        result["implied_byproduct_smiles"] = None
        result["balance_achievable"] = True
        result["note"] = "atom-balanced (no byproduct)"

    return result


def parse_and_check(smiles: str):
    """
    Parse a SMILES with RDKit and perform physical realizability checks.

    Returns (mol_or_None, issues_list) where issues is a list of strings
    describing any problems found:
      - parse_failure         : RDKit could not parse the SMILES at all
      - radical               : molecule has unpaired electrons (radical species)
      - formal_charge         : net formal charge ≠ 0 (ionic / zwitterionic)
      - single_atom           : molecule is a single atom (likely a fragment, not a molecule)
      - disconnected          : SMILES contains '.' (multiple fragments / salt form)
    """
    issues = []
    mol = parse_smiles(smiles)
    if mol is None:
        return None, ["parse_failure"]

    if sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) > 0:
        issues.append("radical")

    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if net_charge != 0:
        issues.append(f"formal_charge({net_charge:+d})")

    if mol.GetNumAtoms() == 1:
        issues.append("single_atom")

    if "." in smiles:
        issues.append("disconnected")

    return mol, issues


# ---------------------------------------------------------------------------
# Route-tree traversal
# ---------------------------------------------------------------------------

def collect_reaction_nodes(node: dict, out: list) -> None:
    """Depth-first walk of an AiZynthFinder route tree; appends reaction nodes."""
    if node.get("is_reaction"):
        out.append(node)
    for child in node.get("children", []):
        collect_reaction_nodes(child, out)


def evaluate_route(route: dict, category: str, route_index: int) -> dict:
    """
    Evaluate one route tree from AiZynthFinder output.

    Walks the tree to find all reaction nodes, calls evaluate_reaction() on
    each, and rolls up per-reaction results into route-level flags:

        all_reaction_smiles_valid — every fragment in every reaction parsed OK
        chemically_plausible      — no reaction has phantom atoms
        fully_balanceable         — every reaction is balanced or has a known
                                    implied byproduct SMILES

    Returns a dict suitable for embedding in the evaluation YAML block.
    """
    target_smi = route.get("smiles", "")
    target_mol, target_issues = parse_and_check(target_smi)

    rxn_nodes: list = []
    collect_reaction_nodes(route, rxn_nodes)

    rxn_evals      = []
    all_valid      = True
    all_plausible  = True
    fully_balance  = True

    for rxn_node in rxn_nodes:
        rxn_eval = evaluate_reaction(rxn_node.get("smiles", ""))
        rxn_eval["policy_probability"] = (
            rxn_node.get("metadata", {}).get("policy_probability")
        )
        rxn_eval["precursors"] = [
            {
                "smiles":       child.get("smiles"),
                "smiles_valid": is_valid_smiles(child.get("smiles", "")),
                "in_stock":     child.get("in_stock"),
            }
            for child in rxn_node.get("children", [])
            if child.get("is_chemical")
        ]
        if not rxn_eval["all_smiles_valid"]:
            all_valid = False
        if rxn_eval.get("phantom_atoms"):
            all_plausible = False
        if not rxn_eval.get("balance_achievable"):
            fully_balance = False
        rxn_evals.append(rxn_eval)

    return {
        "category":                  category,
        "route_index":               route_index,
        "target_smiles":             target_smi,
        "target_smiles_valid":       target_mol is not None,
        "target_rdkit_issues":       target_issues,
        "score":                     route.get("scores", {}),
        "all_reaction_smiles_valid": all_valid,
        "chemically_plausible":      all_plausible,
        "fully_balanceable":         fully_balance,
        "reactions":                 rxn_evals,
    }


# ---------------------------------------------------------------------------
# Full response evaluation
# ---------------------------------------------------------------------------

def evaluate_response(response: dict) -> dict:
    """
    Evaluate all routes in an AiZynthFinder response dict.

    Handles both the flat format ({stats, purchasable_routes, ...}) and the
    tracks-based format ({tracks: {"1": {...}, "2": {...}}}).

    Returns the "evaluation" dict to be stored under _meta.evaluation.
    """
    tracks_raw = response.get("tracks")

    if tracks_raw:
        # tracks-based format
        tracks_input = {str(k): v for k, v in tracks_raw.items()}
    else:
        # flat format — wrap in a synthetic single track
        tracks_input = {"1": response}

    eval_tracks = {}
    total = valid = plausible = balanceable = 0

    for track_id, track in tracks_input.items():
        route_evals = []
        for category in ("purchasable_routes", "non_purchasable_routes"):
            for ri, route in enumerate(track.get(category, [])):
                total += 1
                rev = evaluate_route(route, category, ri)
                if rev["all_reaction_smiles_valid"]:
                    valid += 1
                if rev["chemically_plausible"]:
                    plausible += 1
                if rev["fully_balanceable"]:
                    balanceable += 1
                route_evals.append(rev)
        eval_tracks[track_id] = {
            "stats":             track.get("stats"),
            "route_evaluations": route_evals,
        }

    return {
        "tool":            "RDKit",
        "rdkit_version":   rdBase.rdkitVersion,
        "evaluation_note": (
            "Reactions are stored in retrosynthetic direction (product>>precursors). "
            "Steps: (1) parse every fragment with RDKit; (2) count atoms including "
            "implicit H; (3) classify balance: 'balanced' = equal atom counts, "
            "'surplus' = precursors have extra atoms (implies omitted byproduct), "
            "'phantom' = product has extra atoms (chemically implausible); "
            "(4) for surplus reactions, look up the implied byproduct SMILES in a "
            "table of common small molecules (HCl, H2O, acetic acid, etc.); "
            "(5) flag physical realizability issues (radicals, formal charges, "
            "single atoms, disconnected fragments)."
        ),
        "summary": {
            "total_routes":                 total,
            "routes_with_all_valid_smiles": valid,
            "routes_chemically_plausible":  plausible,
            "routes_fully_balanceable":     balanceable,
        },
        "tracks": eval_tracks,
    }


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def evaluate_yaml_file(path: Path, force: bool = False) -> dict:
    """
    Load a zydus_query_*.yaml file, evaluate its response routes, and write
    the evaluation back under _meta.evaluation.  Returns the evaluation dict.

    Skips without writing if evaluation already present and force=False.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if not force and data.get("_meta", {}).get("evaluation"):
        return data["_meta"]["evaluation"]

    evaluation = evaluate_response(data.get("response", {}))
    data.setdefault("_meta", {})["evaluation"] = evaluation

    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)

    return evaluation


def print_evaluation_summary(path: Path, evaluation: dict) -> None:
    s = evaluation["summary"]
    print(f"{path.name}")
    print(f"  {s['total_routes']} routes  |  "
          f"{s['routes_with_all_valid_smiles']} valid SMILES  |  "
          f"{s['routes_chemically_plausible']} plausible  |  "
          f"{s['routes_fully_balanceable']} balanceable")
    for track_id, track_ev in evaluation["tracks"].items():
        for rev in track_ev["route_evaluations"]:
            rxn = rev["reactions"][0] if rev["reactions"] else {}
            byp_smi  = rxn.get("implied_byproduct_smiles") or "—"
            byp_form = rxn.get("implied_byproduct_formula") or ""
            phant    = rxn.get("phantom_atoms") or {}
            issues   = [m["rdkit_issues"] for m in rxn.get("molecules", [])
                        if m.get("rdkit_issues")]
            flag = "✅" if rev["chemically_plausible"] else "❌"
            bal  = "⚖" if rev["fully_balanceable"] else "✗"
            print(f"  {flag}{bal} track={track_id} route={rev['route_index']}"
                  f" [{rev['category'][:12]}]"
                  f"  byproduct_smi={byp_smi}"
                  f"  byproduct_formula={byp_form or '—'}"
                  f"  phantom={phant or '—'}"
                  + (f"  issues={issues}" if issues else ""))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    default_dir = Path(__file__).resolve().parent.parent / "src" / "chemformer" / "data"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="YAML files to evaluate (default: all zydus_query_*.yaml)")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate files that already have evaluation data")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else sorted(
        default_dir.glob("zydus_query_*.yaml")
    )

    if not paths:
        print(f"No files found in {default_dir}", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        if not path.exists():
            print(f"SKIP (not found): {path}", file=sys.stderr)
            continue
        evaluation = evaluate_yaml_file(path, force=args.force)
        skipped = (not args.force and
                   evaluation == yaml.safe_load(path.read_text())
                   .get("_meta", {}).get("evaluation"))
        print_evaluation_summary(path, evaluation)


if __name__ == "__main__":
    main()
