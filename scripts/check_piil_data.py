from pathlib import Path
from scipy.io import loadmat

data_dir = Path('data/PIIL_LIGHT_PDE_DATA')

files = [
    "burgers_shock.mat",
    "AC.mat",
    "KdV.mat",
    "NLS.mat",
    "KS.mat",
]

for fname in files:
    path = data_dir / fname
    print("\n" + "=" * 80)
    print(fname)
    print("Exists:", path.exists())

    if path.exists():
        data = loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print("Keys:", keys)

        for k in keys:
            arr = data[k]
            if hasattr(arr, "shape"):
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")