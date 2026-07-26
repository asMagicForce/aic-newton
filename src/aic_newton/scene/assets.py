"""Resolve assets vendored with the AIC Newton repository."""

from pathlib import Path


def repository_root() -> Path:
    """Return the root of the source repository."""
    return Path(__file__).resolve().parents[3]


def mjcf_dir() -> Path:
    """Return the directory containing generated AIC MJCF models."""
    return repository_root() / "assets" / "mjcf"


def visual_model_dir() -> Path:
    """Return the directory containing vendored AIC visual models."""
    return repository_root() / "assets" / "aic_assets" / "models"


def scene_asset_dir() -> Path:
    """Return the directory containing AIC reference data."""
    return repository_root() / "assets" / "scene"
