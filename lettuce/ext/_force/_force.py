from abc import ABC, abstractmethod

__all__ = ['Force']


class Force(ABC):
    @abstractmethod
    def source_term(self, u):
        ...

    @abstractmethod
    def u_eq(self, flow: 'Flow'):
        ...

    @property
    @abstractmethod
    def ueq_scaling_factor(self):
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self) -> 'NativeForce':
        ...
