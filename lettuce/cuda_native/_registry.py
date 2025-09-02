from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Set, Hashable, Optional, Any, cast

__all__ = ['Parameter', 'RegistryList', 'Registry', 'CodeRegistryList', 'ParameterRegistryList']


@dataclass
class Parameter:
    variable_type: str
    variable: str

    def __str__(self):
        return f"{self.variable_type} {self.variable}"


class RegistryList(list):
    dimensions: int
    guards: Set[int]

    def __init__(self, dimensions: int):
        list.__init__(self)
        self.dimensions = dimensions
        self.guards = set()

    def registered(self, guard: Optional[Hashable] = None, cond: bool = True) -> bool:
        if not cond:
            return True

        if guard is not None:
            guard = hash(guard)
            if guard in self.guards:
                return True

        return False

    def register(self, guard: Optional[Hashable] = None, cond: bool = True) -> bool:
        if not cond:
            return False

        if guard is not None:
            guard = hash(guard)
            if guard in self.guards:
                return False
            self.guards.add(guard)

        return True

    def append(self, item: Any, guard: Optional[Hashable] = None, cond: bool = True):
        if self.register(guard, cond):
            list.append(self, item)

    def extend(self, other, guard: Optional[Hashable] = None, cond: bool = True):
        if self.register(guard, cond):
            list.extend(self, other)

    def __str__(self):
        raise NotImplementedError()


class CodeRegistryList(RegistryList):
    def mutable(self, variable_type: str, variable: str, value: str) -> str:
        self.append(f"auto {variable}=static_cast<{variable_type}>({value});", guard=variable)
        return variable

    def variable(self, variable_type: str, variable: str, value: str) -> str:
        self.append(f"const auto {variable}=static_cast<{variable_type}>({value});", guard=variable)
        return variable

    def __str__(self):
        return '\n    '.join(self)


class ParameterRegistryList(RegistryList):
    def __str__(self):
        return ','.join(self)


class Registry:
    _buffers: Dict[str, RegistryList[str]]

    def __init__(self, d: int):
        self._buffers = {
            'python_pre': CodeRegistryList(d),
            'python_post': CodeRegistryList(d),
            'cuda': CodeRegistryList(d),
            'pipes': CodeRegistryList(d),
            'pipe': CodeRegistryList(d),
            'py_values': ParameterRegistryList(d),
            'cuda_parameters': ParameterRegistryList(d),
            'cuda_values': ParameterRegistryList(d),
            'kernel_parameters': ParameterRegistryList(d),
        }

    def cuda_hook(self, py_value: str, cuda_param: Parameter, cond: bool = True, ) -> str:
        self._buffers['py_values'].append(str(py_value), cond=cond, guard=cuda_param.variable)
        self._buffers['cuda_parameters'].append(str(cuda_param), cond=cond, guard=cuda_param.variable)
        return cuda_param.variable

    def kernel_hook(self, cuda_value: str, kernel_param: Parameter, cond: bool = True) -> str:
        self._buffers['cuda_values'].append(str(cuda_value), cond=cond, guard=kernel_param.variable)
        self._buffers['kernel_parameters'].append(str(kernel_param), cond=cond, guard=kernel_param.variable)
        return kernel_param.variable

    @cached_property
    def python_pre(self) -> CodeRegistryList:
        return cast(CodeRegistryList, self._buffers['python_pre'])

    @cached_property
    def python_post(self) -> CodeRegistryList:
        return cast(CodeRegistryList, self._buffers['python_post'])

    @cached_property
    def cuda(self) -> CodeRegistryList:
        return cast(CodeRegistryList, self._buffers['cuda'])

    @cached_property
    def pipes(self) -> CodeRegistryList:
        return cast(CodeRegistryList, self._buffers['pipes'])

    @cached_property
    def pipe(self) -> CodeRegistryList:
        return cast(CodeRegistryList, self._buffers['pipe'])

    def joined_buffer(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self._buffers.items()}
