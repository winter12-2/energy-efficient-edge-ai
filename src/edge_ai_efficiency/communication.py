from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommunicationEstimate:
    clients: int
    rounds: int
    bytes_per_parameter: int
    total_bytes: int

    @property
    def total_megabytes(self) -> float:
        return self.total_bytes / (1024 * 1024)


def estimate_model_update_cost(
    parameter_count: int,
    clients: int = 10,
    rounds: int = 20,
    bytes_per_parameter: int = 4,
) -> CommunicationEstimate:
    total_bytes = parameter_count * clients * rounds * bytes_per_parameter
    return CommunicationEstimate(
        clients=clients,
        rounds=rounds,
        bytes_per_parameter=bytes_per_parameter,
        total_bytes=total_bytes,
    )
