from typing import Optional

from .native_generator import NativeLatticeBase


class LatticeBase:
    # lattice: Lattice  TODO add this type hint after import order is fixed
    use_native: bool

    def __init__(self, lattice, use_native=True):
        self.lattice = lattice
        self.use_native = use_native

    def native_available(self) -> bool:
        return False

    def create_native(self) -> Optional[NativeLatticeBase]:
        return None
