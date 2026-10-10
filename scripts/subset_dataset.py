#!/usr/bin/env python3
"""Create subsets of a filtered dataset by fraction or CWE category.

Usage:
  # Fraction-based subsets
  python scripts/subset_dataset.py fraction INPUT OUTPUT --frac 0.11

  # CWE-category subsets (injection-only, no memory)
  python scripts/subset_dataset.py category INPUT OUTPUT --include injection --exclude memory

  # CWE-category subsets (memory-only, no injection)
  python scripts/subset_dataset.py category INPUT OUTPUT --include memory --exclude injection
"""

import argparse
import json
import random
import sys

# CWE classification
INJECTION_CWES = {
    "CWE-77", "CWE-78", "CWE-79", "CWE-89", "CWE-94", "CWE-95",
    "CWE-117", "CWE-134", "CWE-918",  # SSRF is injection-adjacent
}

MEMORY_CWES = {
    "CWE-119", "CWE-120", "CWE-125", "CWE-190", "CWE-476", "CWE-787",
}


def has_category(vulns, cwe_set):
    return any(v in cwe_set for v in vulns)


def cmd_fraction(args):
    with open(args.input) as f:
        samples = json.load(f)
    n = max(1, int(len(samples) * args.frac))
    random.seed(args.seed)
    subset = random.sample(samples, min(n, len(samples)))
    with open(args.output, "w") as f:
        json.dump(subset, f, indent=4)
    print(f"Sampled {len(subset)}/{len(samples)} ({args.frac:.0%}) -> {args.output}")


def cmd_category(args):
    with open(args.input) as f:
        samples = json.load(f)

    cat_map = {"injection": INJECTION_CWES, "memory": MEMORY_CWES}
    include_set = cat_map.get(args.include) if args.include else None
    exclude_set = cat_map.get(args.exclude) if args.exclude else None

    subset = []
    for s in samples:
        vulns = s.get("vulnerabilities", [])
        if include_set and not has_category(vulns, include_set):
            continue
        if exclude_set and has_category(vulns, exclude_set):
            continue
        subset.append(s)

    with open(args.output, "w") as f:
        json.dump(subset, f, indent=4)
    print(f"Filtered {len(subset)}/{len(samples)} -> {args.output}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_frac = sub.add_parser("fraction")
    p_frac.add_argument("input")
    p_frac.add_argument("output")
    p_frac.add_argument("--frac", type=float, required=True)
    p_frac.add_argument("--seed", type=int, default=42)

    p_cat = sub.add_parser("category")
    p_cat.add_argument("input")
    p_cat.add_argument("output")
    p_cat.add_argument("--include", choices=["injection", "memory"])
    p_cat.add_argument("--exclude", choices=["injection", "memory"])

    args = parser.parse_args()
    if args.cmd == "fraction":
        cmd_fraction(args)
    elif args.cmd == "category":
        cmd_category(args)


if __name__ == "__main__":
    main()
