from abc import abstractmethod, ABC

__all__ = [
    'NativeBoundary',
]


class NativeBoundary(ABC):

    def __init__(self, index: int):
        self.index = index

    @staticmethod
    @abstractmethod
    def create(index: int):
        ...
