from abc import ABC, abstractmethod
from lettuce.util import Identifiable


class NativeLettuceBase(Identifiable, ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NativePipelineStep(NativeLettuceBase, ABC):

    def generate_private_parameter(self, name: str):
        return f"{name}_{hex(self.id)}"

    @abstractmethod
    def generate(self, generator: 'Generator'):
        ...
