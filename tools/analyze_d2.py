#!/usr/bin/env python3
"""Analyze mahalanobis_d2 distribution from tracking output.

Usage:
    python3 tools/analyze_d2.py [--csv runs/track/xxx/xxx_tracks.csv]

Output:
    - Sample count
    - Percentiles (50th, 90th, 95th, 99th, max)
    - What % of frames each threshold would reject
"""

import csv, sys, glob

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            v = r.get("mahalanobis_d2", "")
            if v and v != "None" and v.strip():
                try:
                    rows.append(float(v))
                except ValueError:
                    pass
    return rows

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        candidates = glob.glob("runs/track/*/videoplayback_*_tracks.csv")
        if candidates:
            path = candidates[-1]  # latest
    if not path:
        print("Usage: python3 tools/analyze_d2.py [--csv PATH]")
        sys.exit(1)
    
    rows = load(path)
    rows.sort()
    n = len(rows)
    
    print(f"Source: {path}")
    print(f"Samples: {n}")
    print()
    
    if n == 0:
        print("No mahalanobis_d2 values found (all None).")
        return
    
    print("Percentiles:")
    for p in [50, 90, 95, 99, 99.9]:
        idx = min(int(n * p / 100), n - 1)
        print(f"  {p:>5}th: {rows[idx]:>12.1f}")
    print(f"  max  : {rows[-1]:>12.1f}")
    
    print("\nRejection rate at threshold:")
    print(f"  {'threshold':>10} {'reject':>7} {'%':>7}")
    for t in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
        rejected = sum(1 for v in rows if v > t)
        pct = 100.0 * rejected / n
        print(f"  {t:>10} {rejected:>7} {pct:>6.1f}%")

if __name__ == "__main__":
    main()
