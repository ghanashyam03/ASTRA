from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPICE_DIR = DATA_DIR / "spice_kernels"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def spice_dir() -> Path:
    return SPICE_DIR
