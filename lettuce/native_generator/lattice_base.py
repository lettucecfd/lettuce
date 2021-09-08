from typing import Optional


class NativeLatticeBase:
    _name: Optional[str] = None

    @property
    def name(self):
        return self._name
