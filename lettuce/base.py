from typing import Optional


class LatticeBase:
    """Class LatticeBase

    LatticeBase is the Base Class for all Lattice Components.
    LatticeBase is mainly used to make native implementations
    available to all Lattice Components in an uniform way.

    Sub Classes can overwrite `create_native` and create a valid
    native generator. If done so `native_available` should also
    be overwritten to return True.
    """

    lattice: 'Lattice'

    def __init__(self, lattice):
        """Trivial Constructor for Class LatticeBase

        Parameters
        ----------
        lattice: Lattice
            The associated lattice object.
            Used for configuration and to access other lattice components.
        """

        self.lattice = lattice

    def native_available(self) -> bool:
        """native_available

        Returns
        -------
        Whether a native generator is available.
        If so create native should return a valid native generator.
        """
        return False

    def create_native(self) -> Optional['NativeLatticeBase']:
        """create_native

        Returns
        -------
        A native generator that can be used to create a native implementation
        of this component. Native components are generally more performant than
        the default components but native components need to be compiled in the
        first place.
        Check `native_available` before using this method as not every component
        is guaranteed to provide a native generator.
        """
        return None
