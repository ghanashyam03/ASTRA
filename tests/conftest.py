import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPICE_DIR = DATA_DIR / "spice_kernels"

@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT

@pytest.fixture(scope="session")
def spice_dir():
    return SPICE_DIR
