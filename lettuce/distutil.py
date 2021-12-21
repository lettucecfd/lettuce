
import os
from typing import Union
from dataclasses import dataclass
import torch
from torch.types import _int, _size


@dataclass(init=False)
class MPIConfig:
    active: bool
    rank: int = 0
    size: int = 0

    def __init__(self):
        try:
            self.size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.active = True
        except KeyError:
            self.active = False


def roll(input: torch.Tensor, shifts: Union[_int, _size], dims: Union[_int, _size]=(), mpi_config=None) -> torch.Tensor:
    if mpi_config is None:
        return torch.roll(input, shifts, dims)

    assert shifts.abs().max() <= 1

    indices, target_ranks = mpi_config.outgoing(input, shifts)
    indices, source_ranks = mpi_config.incoming(input, shifts)


class MPIConfig:

