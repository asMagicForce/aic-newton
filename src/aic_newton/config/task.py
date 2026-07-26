"""Configure the cable extraction and NIC insertion target."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTargetConfig:
    """Select zero-based scene targets for the automatic task."""

    cable_index: int = 0
    nic_card_index: int = 0
    nic_port_index: int = 0

    def __post_init__(self) -> None:
        """Reject negative scene target indices."""
        for name in ("cable_index", "nic_card_index", "nic_port_index"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def cable_name(self) -> str:
        """Return the selected cable assembly name."""
        return f"cable_{self.cable_index}"


TASK_TARGET = TaskTargetConfig()
