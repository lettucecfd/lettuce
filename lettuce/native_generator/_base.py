from abc import ABC, abstractmethod


class NativeLatticeBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...
