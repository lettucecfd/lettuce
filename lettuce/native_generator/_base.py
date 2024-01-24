from abc import ABC, abstractmethod
from lettuce.util import Identifiable


class NativeLettuceBase(Identifiable, ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...


class NativePipelineStep(NativeLettuceBase, ABC):

    def generate_private_parameter(self, generator: 'Generator', name: str, dtype: str):
        parameter_name = f"{name}_{hex(self.id)}"

        generator.append_python_wrapper_before_buffer("assert hasattr(simulation.Boundary, 'tau')")
        generator.launcher_hook(parameter_name, f"{dtype} {name}", parameter_name, parameter_name)

        generator.kernel_hook(parameter_name, parameter_name, '', '')

    @abstractmethod
    def generate(self, generator: 'Generator'):
        ...
