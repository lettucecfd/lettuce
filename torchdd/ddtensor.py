
__all__ = ["DDTensor"]

import torch
from typing import Union


class DDTensor(torch.TensorType):
    def __init__(self, *input):
        super().__init__(*input)
        self._domains = []
        self._tensors = []

    @property
    def domains(self):
        return self._domains

    @property
    def tensors(self):
        return self._tensors

    @property
    def local_domains(self):
        pass

    @property
    def local_tensors(self):
        pass

    @property
    def local_shapes(self):
        pass

    def __getitem__(self, index: Union[int, slice]) -> Union[torch.Tensor, "DomainDecomposedTensor"]:
        pass

    # def __getattr__(self, field: str):
    #     local_tensor = list(self.local_tensors)
    #     if isinstance(, Callable):
    #         pass

# # alias
# DDTensor = DomainDecomposedTensor
#
# class DDTensorMethod:
#     def __init__(self, name, sub_tensors):
#         pass
#
# class DDShape:
#

