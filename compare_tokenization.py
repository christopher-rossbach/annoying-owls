import pandas as pd
from pathlib import Path
import sys

results_base = Path("results/Qwen2.5-7B-Instruct")

for subdir in ["subliminal_prompting", "allow_hate_prompting", "base_prompting"]:
    d = results_base / subdir
    if not d.exists():
        continue

    old_files = sorted(d.glob("oldtokenization_*.csv"))
    if not old_files:
        continue

    print(f"\n=== {subdir} ===")
    for old_path in old_files:
        suffix = old_path.name.replace("oldtokenization_", "")
        new_path = d / f"spaceinanimal_{suffix}"
        if not new_path.exists():
            print(f"  MISSING: spaceinanimal_{suffix}")
            continue

        old_df = pd.read_csv(old_path, index_col=0)
        new_df = pd.read_csv(new_path, index_col=0)

        diff = (old_df - new_df).abs()
        max_diff = diff.max().max()
        mean_diff = diff.mean().mean()

        status = "OK" if max_diff < 0.1 else "DIFF"
        print(f"  {status} {suffix:<45} max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")
