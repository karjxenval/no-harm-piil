from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import requests
from tqdm import tqdm


DATASETS = {
    # Very useful first set
    "burgers_shock": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat",
        "filename": "burgers_shock.mat",
        "group": "tiny",
        "use": "Burgers inverse viscosity / shock stress test",
    },
    "allen_cahn": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/AC.mat",
        "filename": "AC.mat",
        "group": "tiny",
        "use": "Allen-Cahn reaction-diffusion inverse dynamics",
    },
    "kdv": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/KdV.mat",
        "filename": "KdV.mat",
        "group": "tiny",
        "use": "KdV PDE discovery / coefficient recovery",
    },
    "nls": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/NLS.mat",
        "filename": "NLS.mat",
        "group": "tiny",
        "use": "Nonlinear Schrödinger wave inverse validation",
    },
    "ks": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/KS.mat",
        "filename": "KS.mat",
        "group": "tiny",
        "use": "Kuramoto-Sivashinsky unstable dynamics test",
    },

    # Optional fluid datasets; script will skip them if too large.
    "cylinder_vorticity": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_t0_vorticity.mat",
        "filename": "cylinder_nektar_t0_vorticity.mat",
        "group": "fluid",
        "use": "Cylinder wake vorticity snapshot",
    },
    "cylinder_wake": {
        "url": "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat",
        "filename": "cylinder_nektar_wake.mat",
        "group": "fluid",
        "use": "Cylinder wake fluid-flow validation",
    },
}


SET_MAP = {
    "tiny": ["burgers_shock", "allen_cahn", "kdv", "nls", "ks"],
    "fluid": ["cylinder_vorticity", "cylinder_wake"],
    "all": list(DATASETS.keys()),
}


def get_remote_size_mb(url: str, timeout: int = 20) -> float | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        size = response.headers.get("content-length")
        if size is None:
            return None
        return int(size) / (1024 * 1024)
    except requests.RequestException:
        return None


def download_file(url: str, output_path: Path, timeout: int = 60) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=r"data/PIIL_LIGHT_PDE_DATA",
        help="Output folder.",
    )
    parser.add_argument(
        "--set",
        choices=["tiny", "fluid", "all"],
        default="tiny",
        help="Dataset set to download.",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=150.0,
        help="Skip any single file larger than this size.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check file sizes; do not download.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = SET_MAP[args.set]
    manifest_path = out_dir / "manifest.csv"

    rows = []

    print(f"\nOutput folder: {out_dir}")
    print(f"Dataset set:   {args.set}")
    print(f"Max file size: {args.max_mb:.1f} MB")
    print(f"Check only:    {args.check_only}\n")

    for key in selected:
        item = DATASETS[key]
        url = item["url"]
        filename = item["filename"]
        output_path = out_dir / filename

        print("=" * 80)
        print(f"Dataset: {key}")
        print(f"Use:     {item['use']}")
        print(f"File:    {filename}")

        size_mb = get_remote_size_mb(url)
        if size_mb is None:
            print("Remote size: unknown")
        else:
            print(f"Remote size: {size_mb:.2f} MB")

        status = "planned"

        if size_mb is not None and size_mb > args.max_mb:
            print(f"Skipping because file is larger than {args.max_mb:.1f} MB.")
            status = "skipped_too_large"
        elif output_path.exists():
            print("Already exists. Skipping download.")
            status = "already_exists"
        elif args.check_only:
            print("Check-only mode. Not downloading.")
            status = "checked_only"
        else:
            try:
                print("Downloading...")
                download_file(url, output_path)
                print("Done.")
                status = "downloaded"
            except requests.HTTPError as e:
                print(f"HTTP error: {e}")
                status = "failed_http"
            except requests.RequestException as e:
                print(f"Download error: {e}")
                status = "failed_request"

        rows.append(
            {
                "key": key,
                "filename": filename,
                "group": item["group"],
                "use": item["use"],
                "url": url,
                "size_mb": "" if size_mb is None else f"{size_mb:.3f}",
                "status": status,
            }
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "filename", "group", "use", "url", "size_mb", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nManifest written to:")
    print(manifest_path)
    print("\nRecommended next step:")
    print("Use these files for PIIL sparse/noisy inverse validation.")
    print("Do not download PDEBench unless you later have unlimited data.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(1)