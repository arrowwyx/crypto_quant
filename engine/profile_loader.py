import os
import yaml
from typing import Dict
from factors.registry import registry


PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "factors", "profiles")


def load_factor_profile(name: str) -> Dict:
    """
    Load a factor profile by name from factors/profiles/{name}.yaml.
    Validate that profile['factor'] matches a registered factor.
    """
    if name not in registry.names():
        raise KeyError(f"Factor '{name}' not found in registry; available: {registry.names()}")

    path = os.path.join(PROFILES_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(path, "r") as f:
        prof = yaml.safe_load(f)

    if not isinstance(prof, dict) or prof.get("factor") != name:
        raise ValueError(f"Profile mismatch: expected factor='{name}', got '{prof.get('factor') if isinstance(prof, dict) else prof}'")

    return prof
