from abc import ABC, abstractmethod


class NativeLettuceBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NativePipelineStep(NativeLettuceBase):
    @abstractmethod
    def generate(self, generator: 'Generator'):
        ...
