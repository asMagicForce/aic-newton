"""Resolve model labels by stable suffix."""


def find_label_index(labels: list[str], suffix: str) -> int:
    """Find a label by its stable suffix."""
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")
