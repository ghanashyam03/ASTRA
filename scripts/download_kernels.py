"""Download required SPICE kernels from NASA NAIF."""
import urllib.request
from pathlib import Path

KERNEL_DIR = Path("data/spice_kernels")
KERNEL_DIR.mkdir(parents=True, exist_ok=True)

KERNELS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "de440.bsp":    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
    "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
}

for name, url in KERNELS.items():
    dest = KERNEL_DIR / name
    if dest.exists():
        print(f"  {name} already exists, skipping.")
        continue
    print(f"  Downloading {name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  {name} saved ({dest.stat().st_size // 1024} KB)")

print("All kernels ready.")
