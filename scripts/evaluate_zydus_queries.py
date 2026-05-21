#!/usr/bin/env python3
"""
evaluate_zydus_queries.py — Submit each compound in scripts/zydus_queries.yaml
to the AiZynthFinder + Chemformer retrosynthesis endpoint and save the full
response as a per-compound YAML file.

Output: src/chemformer/data/zydus_query_{compound_name}.yaml

The endpoint streams whitespace heartbeats (~every 15 s) while the MCTS search
runs, then sends the final JSON body.  We use stream=True with a short per-chunk
read timeout so the connection stays alive across the full search window.

Usage:
    python scripts/evaluate_zydus_queries.py [--force] [--time-limit N]
                                             [--iterations N] [--max-routes N]
                                             [--only NAME [NAME ...]]

Options:
    --force         Overwrite existing output files (default: skip)
    --time-limit N  MCTS search time limit in seconds (default: 300)
    --iterations N  MCTS iteration limit (default: 200)
    --max-routes N  Max routes to return per purchasability category (default: 20)
    --only NAME     Only process compound(s) with this query_name (repeatable)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

API_URL = "https://aizynthfinder-chemformer-jk6up23xdq-uc.a.run.app/retrosynthesis"
REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_YAML = REPO_ROOT / "scripts" / "zydus_queries.yaml"
OUT_DIR = REPO_ROOT / "src" / "chemformer" / "data"

# The server sends a whitespace heartbeat every ~15 s, so 30 s per-chunk
# timeout is safe for any search up to the time_limit.
CONNECT_TIMEOUT = 15
READ_CHUNK_TIMEOUT = 60


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")


def output_path(compound: dict, name_counts: Counter) -> Path:
    name = compound["query_name"]
    slug = slugify(name)
    if name_counts[name] > 1:
        slug = f"{slug}_{compound['pubchem_cid']}"
    return OUT_DIR / f"zydus_query_{slug}.yaml"


def call_api(smiles: str, time_limit: int, iteration_limit: int, max_routes: int) -> tuple:
    """Return (response_dict, elapsed_seconds). Raises on error."""
    payload = {
        "smiles": smiles,
        "time_limit": time_limit,
        "iteration_limit": iteration_limit,
        "max_routes": max_routes,
    }
    t0 = time.monotonic()
    resp = requests.post(
        API_URL,
        json=payload,
        timeout=(CONNECT_TIMEOUT, READ_CHUNK_TIMEOUT),
        stream=True,
    )
    resp.raise_for_status()
    # resp.content blocks until the full streaming body is received
    body = resp.content.decode("utf-8").strip()
    elapsed = time.monotonic() - t0
    return json.loads(body), elapsed


def build_record(compound: dict, response: dict, elapsed: float, request_params: dict) -> dict:
    return {
        "_meta": {
            "query_date": datetime.now(timezone.utc).date().isoformat(),
            "query_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "source": "aizynthfinder-chemformer-jk6up23xdq-uc.a.run.app",
            "compound": compound,
            "request": request_params,
        },
        "response": response,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--time-limit", type=int, default=300, metavar="N")
    parser.add_argument("--iterations", type=int, default=200, metavar="N")
    parser.add_argument("--max-routes", type=int, default=20, metavar="N")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Only process these query_names")
    args = parser.parse_args()

    request_params = {
        "time_limit": args.time_limit,
        "iteration_limit": args.iterations,
        "max_routes": args.max_routes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERIES_YAML) as f:
        data = yaml.safe_load(f)

    molecules = data["molecules"]
    name_counts = Counter(m["query_name"] for m in molecules)

    if args.only:
        only_set = set(args.only)
        molecules = [m for m in molecules if m["query_name"] in only_set]
        if not molecules:
            print(f"No molecules matched --only {args.only}", file=sys.stderr)
            sys.exit(1)

    total = len(molecules)
    print(f"Processing {total} compound(s)  |  output: {OUT_DIR}")
    print(f"Request params: time_limit={args.time_limit}s  iterations={args.iterations}  max_routes={args.max_routes}\n")

    skipped = processed = errors = 0

    for i, compound in enumerate(molecules, 1):
        name = compound["query_name"]
        cid = compound.get("pubchem_cid", "?")
        smiles = (compound.get("smiles") or "").strip()
        mw = compound.get("molecular_weight") or compound.get("complexity", "?")
        out = output_path(compound, name_counts)

        print(f"[{i}/{total}] {name}  CID={cid}  MW={compound.get('molecular_weight', '?')}  complexity={compound.get('complexity', '?')}")

        if not smiles:
            print("  SKIP — no SMILES\n")
            skipped += 1
            continue

        if out.exists() and not args.force:
            print(f"  SKIP — output exists: {out.name}\n")
            skipped += 1
            continue

        print(f"  SMILES: {smiles[:72]}{'...' if len(smiles) > 72 else ''}")
        print(f"  Querying API ...", end="", flush=True)

        try:
            response, elapsed = call_api(smiles, args.time_limit, args.iterations, args.max_routes)
        except requests.exceptions.Timeout:
            print(f"\n  ERROR — timed out (chunk gap > {READ_CHUNK_TIMEOUT}s)\n")
            errors += 1
            continue
        except requests.exceptions.RequestException as exc:
            print(f"\n  ERROR — {exc}\n")
            errors += 1
            continue
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"\n  ERROR — bad JSON: {exc}\n")
            errors += 1
            continue

        # Response is either flat {stats, purchasable_routes, ...} or
        # tracks-based {tracks: {"1": {stats, purchasable_routes, ...}, ...}}
        if "tracks" in response:
            tracks = list(response["tracks"].values())
            stats = tracks[-1].get("stats", {}) if tracks else {}
            n_purchasable = sum(len(t.get("purchasable_routes", [])) for t in tracks)
            n_non_purchasable = sum(len(t.get("non_purchasable_routes", [])) for t in tracks)
        else:
            stats = response.get("stats", {})
            n_purchasable = len(response.get("purchasable_routes", []))
            n_non_purchasable = len(response.get("non_purchasable_routes", []))
        print(
            f" {elapsed:.1f}s  |  "
            f"nodes={stats.get('number_of_nodes', '?')}  "
            f"routes={stats.get('number_of_routes', '?')}  "
            f"solved={stats.get('number_of_solved_routes', '?')}  "
            f"purchasable={n_purchasable}  non_purchasable={n_non_purchasable}"
        )

        record = build_record(compound, response, elapsed, request_params)

        with open(out, "w") as f:
            yaml.dump(record, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        print(f"  saved: {out.name}\n")
        processed += 1

    print(f"Finished: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
