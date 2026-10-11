"""
Download the FORMA persona and vignette data for offline use.

FORMA (github.com/Amit-Oren/FORMA) is the prior-work pipeline that generated
the 500 clinician-validated personas and their vignettes. This simulation
consumes them rather than regenerating personas of its own, so a run needs no
LLM calls to build a patient and `persona_id` acts as the seed.

Two vignette conditions are fetched:
  no_formulation — written without the cognitive graph, so no causal links are
                   narrated. This is what the patient agent reads.
  full           — written with the graph. Kept only as an ablation arm.

Data is CC BY 4.0; code MIT.

    python scripts/fetch_forma_data.py
    python scripts/fetch_forma_data.py --model gpt-5.4 --force
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "Amit-Oren/FORMA"
API = f"https://api.github.com/repos/{REPO}/contents/"
DEST = Path("data/forma")
DEFAULT_MODEL = "claude-sonnet-4-6"


def _listing(path: str) -> list[dict]:
    """One API call per directory — content comes from raw.githubusercontent."""
    url = f"{API}{path}?per_page=100"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sys.exit("GitHub API rate limit hit. Wait an hour, or set GITHUB_TOKEN.")
        raise


def _grab(entry: dict, out_dir: Path) -> tuple[str, bool]:
    target = out_dir / entry["name"]
    if target.exists() and target.stat().st_size == entry["size"]:
        return entry["name"], False
    with urllib.request.urlopen(entry["download_url"], timeout=60) as r:
        target.write_bytes(r.read())
    return entry["name"], True


def fetch(remote: str, local: Path, workers: int = 16) -> None:
    entries = [e for e in _listing(remote) if e["type"] == "file"]
    local.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_grab, e, local) for e in entries]
        for i, fut in enumerate(as_completed(futures), 1):
            _, written = fut.result()
            done += written
            skipped += (not written)
            if i % 50 == 0 or i == len(entries):
                print(f"  {local.as_posix():<44} {i}/{len(entries)}", end="\r")
    size = sum(f.stat().st_size for f in local.glob("*.json")) / 1024 / 1024
    print(f"  {local.as_posix():<44} {len(entries)} files, "
          f"{done} new, {skipped} cached, {size:.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"vignette generator model (default: {DEFAULT_MODEL})")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the `full` condition — only needed for the ablation")
    args = ap.parse_args()

    print(f"Fetching FORMA data -> {DEST.as_posix()}/  (model: {args.model})")
    fetch("data/personas", DEST / "personas")
    fetch(f"data/vignettes/no_formulation/{args.model}",
          DEST / "vignettes" / "no_formulation")
    if not args.skip_full:
        fetch(f"data/vignettes/full/{args.model}", DEST / "vignettes" / "full")

    (DEST / "SOURCE.txt").write_text(
        f"Personas and vignettes from {REPO}\n"
        f"Vignette generator model: {args.model}\n"
        "Data licensed CC BY 4.0. Fetched by scripts/fetch_forma_data.py\n",
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()
