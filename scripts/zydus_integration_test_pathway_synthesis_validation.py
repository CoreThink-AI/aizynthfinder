"""Validate pathway viability by scoring each reaction step with forward synthesis scoring.

Reads the retrosynthesis integration test output YAML and calls the Chemformer
/synthesis/score endpoint for every reaction step in every pathway, computing a
forward log-likelihood score: P(product | reactants) via teacher-forced decoding.

Higher (less negative) scores indicate that the BART model — having learned reaction
chemistry from USPTO-50K — considers the forward reaction more plausible.

Output:
    src/chemformer/data/zydus_synthesis_validation_{hash6}.yaml
    docs/zydus_synthesis_validation_report_{date}.md

Run from the project root:
    python scripts/zydus_integration_test_pathway_synthesis_validation.py [options]

Options:
    --input       Path to retrosynthesis YAML (default: newest matching file)
    --model       Checkpoint stem (default: backward_uspto50k)
    --url         API base URL
    --output-dir  Directory for output files (default: src/chemformer/data)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

# ── defaults ──────────────────────────────────────────────────────────────────

API_URL = "https://chemformer-retrosynthesis-knq67derjq-uc.a.run.app"
DEFAULT_MODEL = "backward_uspto50k"
INPUT_GLOB = "src/chemformer/data/zydus_queries_chemformer_recursive_*.yaml"
OUTPUT_DIR = Path("src/chemformer/data")
DOCS_DIR = Path("docs")

# Timeout: (connect_s, read_s) — HTTP/2 PING frames can defeat a single integer timeout
REQUEST_TIMEOUT = (10, 90)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _score_reactions(smarts_list: list, model: str, url: str) -> list:
    """Call /synthesis/{model}/score; return list of dicts with ll + n_tokens."""
    endpoint = f"{url}/synthesis/{model}/score"
    payload = {"reaction_smarts": smarts_list}
    try:
        resp = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("scores", [])
    except requests.exceptions.Timeout:
        print(f"  [TIMEOUT] scoring {len(smarts_list)} reaction(s)")
        return [{"reaction_smarts": s, "log_likelihood": None, "n_product_tokens": 0} for s in smarts_list]
    except Exception as exc:
        print(f"  [ERROR] scoring: {exc}")
        return [{"reaction_smarts": s, "log_likelihood": None, "n_product_tokens": 0} for s in smarts_list]


# ── scoring thresholds ────────────────────────────────────────────────────────

def _ll_label(ll: float) -> str:
    if ll is None:
        return "n/a"
    if ll > -2.0:
        return "high"
    if ll > -5.0:
        return "moderate"
    return "low"


def _pathway_mean_ll(scored_steps: list) -> float | None:
    lls = [s["forward_ll"] for s in scored_steps if s.get("forward_ll") is not None]
    return round(sum(lls) / len(lls), 4) if lls else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="Path to retrosynthesis YAML")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Checkpoint stem")
    parser.add_argument("--url", default=API_URL, help="API base URL")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    # Resolve input file
    if args.input:
        input_path = Path(args.input)
    else:
        candidates = sorted(Path(".").glob(INPUT_GLOB))
        if not candidates:
            print(f"No retrosynthesis YAML found matching {INPUT_GLOB}", file=sys.stderr)
            sys.exit(1)
        input_path = candidates[-1]
    print(f"Input:  {input_path}")

    retro_data = yaml.safe_load(input_path.read_text())
    retro_meta = retro_data.get("metadata", {})
    results_in = retro_data.get("results", [])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Derive short hash from input filename for output naming
    stem = input_path.stem  # e.g. zydus_queries_chemformer_recursive_db5e90
    parts = stem.split("_")
    short_hash = parts[-1] if len(parts[-1]) == 6 else stem[-6:]

    timestamp = datetime.now(timezone.utc).isoformat()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Model:  {args.model}")
    print(f"URL:    {args.url}")
    print()

    # ── health check ──────────────────────────────────────────────────────────
    try:
        hc = requests.get(f"{args.url}/health", timeout=(5, 10))
        if hc.status_code != 200:
            print(f"Health check failed: {hc.status_code}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── process each molecule ─────────────────────────────────────────────────
    results_out = []
    summary_rows = []  # for report table

    total_reactions = sum(
        len(p.get("steps", []))
        for r in results_in
        for p in r.get("pathways", [])
    )
    print(f"Scoring {total_reactions} reaction steps across {len(results_in)} molecules ...\n")

    for mol in results_in:
        mol_name = mol["query_name"]
        pathways = mol.get("pathways", [])

        if not pathways:
            print(f"  {mol_name}: 0 pathways (skipped)")
            results_out.append({
                "query_name": mol_name,
                "pubchem_cid": mol.get("pubchem_cid"),
                "input_smiles": mol.get("input_smiles"),
                "pathways_scored": [],
                "best_pathway_rank": None,
                "best_pathway_mean_ll": None,
            })
            summary_rows.append((mol_name, "—", "—", "—", "timeout/OOD"))
            continue

        scored_pathways = []
        for pathway in pathways:
            steps = pathway.get("steps", [])
            if not steps:
                scored_pathways.append({
                    "pathway_rank": pathway.get("pathway_rank"),
                    "step_count": 0,
                    "scored_steps": [],
                    "pathway_mean_ll": None,
                    "pathway_ll_label": "n/a",
                    "retro_min_ll": pathway.get("min_log_likelihood"),
                    "all_leaves_found": pathway.get("all_leaves_found"),
                    "all_leaves_purchasable": pathway.get("all_leaves_purchasable"),
                })
                continue

            smarts_list = [s["reaction_smarts"] for s in steps]
            t0 = time.time()
            scores = _score_reactions(smarts_list, args.model, args.url)
            elapsed = round(time.time() - t0, 1)

            # Merge scores back with step metadata
            scored_steps = []
            score_map = {s["reaction_smarts"]: s for s in scores}
            for step in steps:
                smarts = step["reaction_smarts"]
                score = score_map.get(smarts, {})
                forward_ll = score.get("log_likelihood")
                scored_steps.append({
                    "step": step["step"],
                    "reaction_smarts": smarts,
                    "product_smiles": step.get("product_smiles"),
                    "retro_ll": step.get("log_likelihood"),
                    "forward_ll": forward_ll,
                    "forward_ll_label": _ll_label(forward_ll),
                    "n_product_tokens": score.get("n_product_tokens", 0),
                })

            mean_ll = _pathway_mean_ll(scored_steps)
            scored_pathways.append({
                "pathway_rank": pathway.get("pathway_rank"),
                "step_count": len(steps),
                "scored_steps": scored_steps,
                "pathway_mean_ll": mean_ll,
                "pathway_ll_label": _ll_label(mean_ll),
                "retro_min_ll": pathway.get("min_log_likelihood"),
                "all_leaves_found": pathway.get("all_leaves_found"),
                "all_leaves_purchasable": pathway.get("all_leaves_purchasable"),
                "latency_s": elapsed,
            })

            label = _ll_label(mean_ll)
            ll_str = f"{mean_ll:.3f}" if mean_ll is not None else "n/a"
            print(f"  {mol_name} pathway {pathway['pathway_rank']}: "
                  f"mean_fwd_ll={ll_str} ({label}), {len(steps)} step(s), {elapsed}s")

        # Pick best pathway by mean forward ll
        valid = [p for p in scored_pathways if p.get("pathway_mean_ll") is not None]
        best = max(valid, key=lambda p: p["pathway_mean_ll"]) if valid else None

        results_out.append({
            "query_name": mol_name,
            "pubchem_cid": mol.get("pubchem_cid"),
            "input_smiles": mol.get("input_smiles"),
            "pathways_scored": scored_pathways,
            "best_pathway_rank": best["pathway_rank"] if best else None,
            "best_pathway_mean_ll": best["pathway_mean_ll"] if best else None,
        })

        best_ll_str = f"{best['pathway_mean_ll']:.3f}" if best else "n/a"
        best_rank = best["pathway_rank"] if best else "—"
        summary_rows.append((
            mol_name,
            str(len(pathways)),
            best_rank,
            best_ll_str,
            _ll_label(best["pathway_mean_ll"]) if best else "n/a",
        ))
        print()

    # ── write YAML output ─────────────────────────────────────────────────────
    output = {
        "metadata": {
            "input_file": str(input_path),
            "retro_model": retro_meta.get("model"),
            "scoring_model": args.model,
            "api_url": args.url,
            "timestamp": timestamp,
        },
        "results": results_out,
    }
    yaml_path = output_dir / f"zydus_synthesis_validation_{short_hash}.yaml"
    yaml_path.write_text(yaml.dump(output, sort_keys=False, allow_unicode=True, width=120))
    print(f"YAML output: {yaml_path}")

    # ── write Markdown report ─────────────────────────────────────────────────
    report_path = DOCS_DIR / f"zydus_synthesis_validation_report_{date_str}.md"
    _write_report(report_path, summary_rows, results_out, retro_meta, args, timestamp, short_hash)
    print(f"Report:      {report_path}")


# ── report generation ─────────────────────────────────────────────────────────

def _write_report(path, summary_rows, results_out, retro_meta, args, timestamp, short_hash):
    lines = []
    lines.append("# Chemformer Synthesis Viability Validation Report\n")
    lines.append(f"**Date:** {timestamp[:10]}  ")
    lines.append(f"**Retrosynthesis model:** `{retro_meta.get('model', '?')}`  ")
    lines.append(f"**Scoring model:** `{args.model}`  ")
    lines.append(f"**Service:** `{args.url}`  ")
    lines.append(f"**Input:** `zydus_queries_chemformer_recursive_{short_hash}.yaml`\n")
    lines.append("---\n")
    lines.append("## Summary\n")
    lines.append("Forward log-likelihood scores P(product | reactants) via teacher-forced decoding.")
    lines.append("Higher (less negative) = model considers the forward reaction more plausible.\n")
    lines.append("| Molecule | Pathways | Best rank | Best mean fwd ll | Viability |")
    lines.append("|---|---|---|---|---|")
    for row in summary_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    lines.append("")
    lines.append("---\n")
    lines.append("## Per-Molecule Detail\n")

    for mol in results_out:
        mol_name = mol["query_name"]
        lines.append(f"### {mol_name}\n")
        pathways = mol.get("pathways_scored", [])
        if not pathways:
            lines.append("No pathways found (timeout or OOD molecule).\n")
            continue

        for pw in pathways:
            rank = pw.get("pathway_rank")
            mean_ll = pw.get("pathway_mean_ll")
            label = pw.get("pathway_ll_label", "n/a")
            purchasable = "yes" if pw.get("all_leaves_purchasable") else "no"
            retro_ll = pw.get("retro_min_ll")

            ll_str = f"{mean_ll:.4f}" if mean_ll is not None else "n/a"
            retro_str = f"{retro_ll:.4f}" if retro_ll is not None else "n/a"
            lines.append(f"**Pathway {rank}** — mean fwd ll={ll_str} ({label}), "
                         f"retro min ll={retro_str}, purchasable={purchasable}\n")

            steps = pw.get("scored_steps", [])
            if steps:
                lines.append("| Step | Reaction SMARTS | Retro ll | Fwd ll | Viability |")
                lines.append("|---|---|---|---|---|")
                for s in steps:
                    smarts = s["reaction_smarts"]
                    if len(smarts) > 80:
                        smarts = smarts[:77] + "..."
                    retro_s = f"{s['retro_ll']:.4f}" if s.get("retro_ll") is not None else "n/a"
                    fwd_s = f"{s['forward_ll']:.4f}" if s.get("forward_ll") is not None else "n/a"
                    lines.append(f"| {s['step']} | `{smarts}` | {retro_s} | {fwd_s} | {s['forward_ll_label']} |")
                lines.append("")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
