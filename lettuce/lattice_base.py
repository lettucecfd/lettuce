from typing import Optional
from . import *
from .native_generator.lattice_base import NativeLatticeBase


class LatticeBase:
    lattice: 'Lattice'
    use_native: bool

    def __init__(self, lattice: 'Lattice', use_native: bool = True):
        self.lattice = lattice
        self.use_native = use_native

    def native_available(self) -> bool:
        return False

    def create_native(self) -> Optional[NativeLatticeBase]:
        return None
