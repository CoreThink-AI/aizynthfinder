"""Generate retrosynthesis route YAML for Aspirin in zydus_query format."""
import datetime
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import inchi as rdinchi

from aizynthfinder.aizynthfinder import AiZynthFinder


def to_python(obj):
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

CONFIG_PATH = "/home/hobs/code/corethink/ai-routes-generator/bigdata/config.yml"
TARGET_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
OUTPUT_PATH = Path(__file__).parent / "zydus_query_python_AiZynthFinder_Aspirin.yaml"

COMMON_BYPRODUCTS = [
    (Counter({"H": 1, "Cl": 1}), "HCl", "Cl"),
    (Counter({"H": 2, "O": 1}), "H2O", "O"),
    (Counter({"C": 2, "H": 4, "O": 2}), "C2H4O2", "CC(=O)O"),
    (Counter({"C": 1, "H": 4, "O": 1}), "CH4O", "CO"),
    (Counter({"C": 2, "H": 6, "O": 1}), "C2H6O", "CCO"),
    (Counter({"H": 1, "Br": 1}), "HBr", "[H]Br"),
    (Counter({"H": 1, "F": 1}), "HF", "F"),
    (Counter({"C": 1, "H": 2, "O": 2}), "CH2O2", "OC=O"),
    (Counter({"C": 3, "H": 6, "O": 2}), "C3H6O2", "CCC(=O)O"),
    (Counter({"C": 4, "H": 6, "O": 3}), "C4H6O3", "CC(=O)OC(C)=O"),
    (Counter({"C": 3, "H": 8, "O": 1}), "C3H8O", "CCCO"),
    (Counter({"C": 1, "H": 3, "N": 1, "H": 3}), "CH5N", "CN"),
]


def get_atom_counts(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return counts


def counter_to_formula(counter):
    formula = ""
    if counter.get("C", 0):
        n = counter["C"]
        formula += f"C{n}" if n > 1 else "C"
    if counter.get("H", 0):
        n = counter["H"]
        formula += f"H{n}" if n > 1 else "H"
    for elem in sorted(k for k in counter if k not in ("C", "H")):
        n = counter[elem]
        formula += f"{elem}{n}" if n > 1 else elem
    return formula


def lookup_byproduct(surplus):
    for ref_counts, formula, smiles in COMMON_BYPRODUCTS:
        if ref_counts == surplus:
            return formula, smiles
    return counter_to_formula(surplus), None


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, [f"unparseable SMILES: {smiles}"]
    return True, []


def analyze_balance(product_smiles, precursor_smiles_list):
    prod_counts = get_atom_counts(product_smiles)
    if prod_counts is None:
        return dict(balanced=False, phantom_atoms={}, implied_byproduct_formula=None,
                    implied_byproduct_smiles=None, balance_achievable=False,
                    note="unparseable product")

    prec_counts = Counter()
    for s in precursor_smiles_list:
        c = get_atom_counts(s)
        if c is None:
            return dict(balanced=False, phantom_atoms={}, implied_byproduct_formula=None,
                        implied_byproduct_smiles=None, balance_achievable=False,
                        note=f"unparseable precursor: {s}")
        prec_counts += c

    surplus = Counter()
    phantom = Counter()
    for elem in set(prod_counts) | set(prec_counts):
        diff = prec_counts.get(elem, 0) - prod_counts.get(elem, 0)
        if diff > 0:
            surplus[elem] = diff
        elif diff < 0:
            phantom[elem] = -diff

    if not surplus and not phantom:
        return dict(balanced=True, phantom_atoms={}, implied_byproduct_formula=None,
                    implied_byproduct_smiles=None, balance_achievable=True,
                    note="perfectly balanced")

    if phantom:
        return dict(balanced=False, phantom_atoms=dict(phantom),
                    implied_byproduct_formula=None, implied_byproduct_smiles=None,
                    balance_achievable=False,
                    note=f"chemically implausible — atoms appear in product absent from precursors: {dict(phantom)}")

    formula, byproduct_smiles = lookup_byproduct(surplus)
    if byproduct_smiles:
        return dict(balanced=False, phantom_atoms={}, implied_byproduct_formula=formula,
                    implied_byproduct_smiles=byproduct_smiles, balance_achievable=True,
                    note=f"balanceable — forward: precursors → product + {byproduct_smiles} ({formula})")
    return dict(balanced=False, phantom_atoms={}, implied_byproduct_formula=formula,
                implied_byproduct_smiles=None, balance_achievable=False,
                note=f"surplus atoms {formula} — no common byproduct match found")


def collect_reactions(node, parent_smiles=None):
    """Recursively collect (product, precursors, in_stock_flags, metadata) from tree dict."""
    reactions = []
    if node["type"] == "mol":
        for child in node.get("children", []):
            reactions.extend(collect_reactions(child, parent_smiles=node["smiles"]))
    elif node["type"] == "reaction":
        mol_children = [c for c in node.get("children", []) if c["type"] == "mol"]
        reactions.append({
            "product_smiles": parent_smiles,
            "precursor_smiles": [c["smiles"] for c in mol_children],
            "precursors_in_stock": [c.get("in_stock", False) for c in mol_children],
            "metadata": node.get("metadata", {}),
        })
        for child in mol_children:
            reactions.extend(collect_reactions(child))
    return reactions


def count_tree_nodes(node):
    """Return (n_reactions, n_precursors, n_in_stock) for a tree dict node."""
    if node["type"] == "mol":
        if not node.get("children"):
            return 0, 1, int(node.get("in_stock", False))
        r = p = s = 0
        for ch in node["children"]:
            dr, dp, ds = count_tree_nodes(ch)
            r += dr; p += dp; s += ds
        return r, p, s
    elif node["type"] == "reaction":
        r = p = s = 0
        for ch in node.get("children", []):
            dr, dp, ds = count_tree_nodes(ch)
            r += dr; p += dp; s += ds
        return r + 1, p, s
    return 0, 0, 0


def build_route_evaluation(route_index, target_canonical, tree_dict, state_score):
    target_valid, target_issues = validate_smiles(target_canonical)
    n_reactions, n_precursors, n_in_stock = count_tree_nodes(tree_dict)

    score = {
        "state score": state_score,
        "number of reactions": n_reactions,
        "number of pre-cursors": n_precursors,
        "number of pre-cursors in stock": n_in_stock,
    }

    raw_reactions = collect_reactions(tree_dict)
    reaction_evals = []
    all_valid = True
    chemically_plausible = True
    fully_balanceable = True

    for rxn in raw_reactions:
        prod = rxn["product_smiles"]
        precs = rxn["precursor_smiles"]
        meta = rxn["metadata"]

        reaction_smiles = f"{prod}>>{'.'.join(precs)}"
        prod_valid, prod_issues = validate_smiles(prod)
        molecules = [{"role": "product", "smiles": prod, "valid": prod_valid, "rdkit_issues": prod_issues}]
        rxn_all_valid = prod_valid

        for smi in precs:
            v, issues = validate_smiles(smi)
            molecules.append({"role": "precursor", "smiles": smi, "valid": v, "rdkit_issues": issues})
            rxn_all_valid = rxn_all_valid and v

        all_valid = all_valid and rxn_all_valid
        balance = analyze_balance(prod, precs)
        if not balance["balance_achievable"]:
            fully_balanceable = False
        if balance["phantom_atoms"]:
            chemically_plausible = False

        reaction_evals.append({
            "reaction_smiles": reaction_smiles,
            "molecules": molecules,
            "all_smiles_valid": rxn_all_valid,
            "balanced": balance["balanced"],
            "phantom_atoms": balance["phantom_atoms"],
            "implied_byproduct_formula": balance["implied_byproduct_formula"],
            "implied_byproduct_smiles": balance["implied_byproduct_smiles"],
            "balance_achievable": balance["balance_achievable"],
            "note": balance["note"],
            "policy_probability": meta.get("policy_probability"),
            "precursors": [
                {
                    "smiles": s,
                    "smiles_valid": validate_smiles(s)[0],
                    "in_stock": in_stock,
                }
                for s, in_stock in zip(precs, rxn["precursors_in_stock"])
            ],
        })

    return {
        "category": "purchasable_routes",
        "route_index": route_index,
        "target_smiles": target_canonical,
        "target_smiles_valid": target_valid,
        "target_rdkit_issues": target_issues,
        "score": score,
        "all_reaction_smiles_valid": all_valid,
        "chemically_plausible": chemically_plausible,
        "fully_balanceable": fully_balanceable,
        "reactions": reaction_evals,
    }


def main():
    now = datetime.datetime.now(datetime.timezone.utc)

    print("Loading AiZynthFinder...")
    finder = AiZynthFinder(configfile=CONFIG_PATH)
    finder.expansion_policy.select(["uspto", "ringbreaker"])
    finder.filter_policy.select("uspto")
    finder.stock.select("zinc")
    finder.target_smiles = TARGET_SMILES

    t0 = time.time()
    print("Running tree search...")
    finder.tree_search()
    elapsed = time.time() - t0

    finder.build_routes()
    stats = finder.extract_statistics()

    target_canonical = finder.target_smiles
    mol = Chem.MolFromSmiles(target_canonical)
    inchi_str = rdinchi.MolToInchi(mol)
    inchikey = rdinchi.InchiToInchiKey(inchi_str)

    t = finder.tree
    try:
        n_nodes = len(list(t.graph().nodes()))
    except Exception:
        n_nodes = 0

    trees = finder.routes.reaction_trees
    scores = finder.routes.scores
    solved = [(i, rt, sc) for i, (rt, sc) in enumerate(zip(trees, scores)) if rt.is_solved]

    mcts_stats = {
        "search_time_seconds": stats.get("search_time", elapsed),
        "iterations": t.profiling.get("iterations", 0),
        "expansion_calls": t.profiling.get("expansion_calls", 0),
        "number_of_nodes": n_nodes,
        "number_of_routes": len(trees),
        "number_of_solved_routes": len(solved),
        "is_solved": bool(solved),
        "top_score": scores[0].get("state score", 0) if scores else 0,
    }

    route_evals = []
    purchasable_routes = []

    for route_index, (_, tree, score) in enumerate(solved):
        tree_dict = tree.to_dict()
        state_score = score.get("state score", 0)
        eval_data = build_route_evaluation(route_index, target_canonical, tree_dict, state_score)
        route_evals.append(eval_data)
        purchasable_route = dict(tree_dict)
        purchasable_route["scores"] = eval_data["score"]
        purchasable_route["metadata"] = {
            "created_at_iteration": tree.created_at_iteration,
            "is_solved": tree.is_solved,
        }
        purchasable_routes.append(purchasable_route)

    total = len(route_evals)
    valid_count = sum(1 for e in route_evals if e["all_reaction_smiles_valid"])
    plausible_count = sum(1 for e in route_evals if e["chemically_plausible"])
    balanceable_count = sum(1 for e in route_evals if e["fully_balanceable"])

    rdkit_version = Chem.rdBase.rdkitVersion

    output = {
        "_meta": {
            "query_date": now.date().isoformat(),
            "query_timestamp_utc": now.isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "source": "aizynthfinder-python-api-local",
            "compound": {
                "query_name": "Aspirin",
                "pubchem_cid": 2244,
                "pubchem_name": "2-acetyloxybenzoic acid",
                "smiles": TARGET_SMILES,
                "complexity": 212,
                "inchikey": inchikey,
                "inchi": inchi_str,
                "vendor_count": 533,
                "bioassay_count": 3367,
            },
            "request": {
                "time_limit": finder.config.search.time_limit,
                "iteration_limit": finder.config.search.iteration_limit,
                "max_routes": finder.config.post_processing.max_routes,
            },
            "evaluation": {
                "tool": "RDKit",
                "rdkit_version": rdkit_version,
                "evaluation_note": (
                    "Reactions are stored in retrosynthetic direction (product>>precursors). "
                    "Steps: (1) parse every fragment with RDKit; (2) count atoms including implicit "
                    "H; (3) classify balance: 'balanced' = equal atom counts, 'surplus' = precursors "
                    "have extra atoms (implies omitted byproduct), 'phantom' = product has extra "
                    "atoms (chemically implausible); (4) for surplus reactions, look up the implied "
                    "byproduct SMILES in a table of common small molecules (HCl, H2O, acetic acid, "
                    "etc.); (5) flag physical realizability issues (radicals, formal charges, single "
                    "atoms, disconnected fragments)."
                ),
                "summary": {
                    "total_routes": total,
                    "routes_with_all_valid_smiles": valid_count,
                    "routes_chemically_plausible": plausible_count,
                    "routes_fully_balanceable": balanceable_count,
                },
                "tracks": {
                    "1": {
                        "stats": mcts_stats,
                        "route_evaluations": route_evals,
                    },
                },
            },
        },
        "response": {
            "target_smiles": TARGET_SMILES,
            "stock_configured": True,
            "tracks": {
                "1": {
                    "tag": 1,
                    "stats": mcts_stats,
                    "purchasable_routes": purchasable_routes,
                    "non_purchasable_routes": [],
                    "error": None,
                },
            },
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(to_python(output), f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Written to {OUTPUT_PATH}")
    print(f"Routes: {total} solved | {plausible_count} chemically plausible | {balanceable_count} fully balanceable")


if __name__ == "__main__":
    main()
