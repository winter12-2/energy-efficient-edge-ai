from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommunicationEstimate:

    # stores the estimation cost.
    clients: int
    rounds: int
    bytes_per_parameter: int
    total_bytes: int

    @property
    def total_megabytes(self) -> float:

        # converting bytes to megabytes
        return self.total_bytes / (1024 * 1024)


def estimate_model_update_cost(
    parameter_count: int,
    clients: int = 10,
    rounds: int = 20,
    bytes_per_parameter: int = 4,
) -> CommunicationEstimate:
    
    # estimate how much data is sent during model updates
    # Formula: total bytes = parameters * clients * rounds * bytes per parameter
    # eg: float32 uses 4 bytes per parameter, int8 uses 1 byte per parameter

    if parameter_count < 0:
        raise ValueError("parameter_count cannot be negative")
    if clients <= 0:
        raise ValueError("clients must be greater than 0")
    if rounds <= 0:
        raise ValueError("rounds must be greater than 0")
    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be greater than 0")
    
    total_bytes = parameter_count * clients * rounds * bytes_per_parameter
    return CommunicationEstimate(
        clients=clients,
        rounds=rounds,
        bytes_per_parameter=bytes_per_parameter,
        total_bytes=total_bytes,
    )
