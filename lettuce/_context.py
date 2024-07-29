import torch
import numpy as np

from typing import List, Optional, Union

__all__ = ['Context']


class Context:
    def __init__(self, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 use_native: Optional[bool] = None):

        # check sanity of context configuration

        if device is None and use_native is None:
            device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            use_native = True if torch.cuda.is_available() else False

        elif device is None:
            assert not use_native or torch.cuda.is_available(), \
                ('native extension explicitly requested but cuda is not '
                 'available!')
            device = torch.device('cuda:0')

        elif use_native is None:
            if 'cuda' in str(device):
                assert torch.cuda.is_available(), ('cuda device explicitly '
                                                   'requested but cuda is not '
                                                   'available!')
                use_native = True
            else:
                assert 'cpu' in str(
                    device), (f"lettuce is designed to work on cpu or cuda "
                              f"devices. {device} is not supported!")
                use_native = False

        else:
            if 'cuda' in str(device):
                assert torch.cuda.is_available(), ('cuda device explicitly '
                                                   'requested but cuda is not '
                                                   'available!')
            else:
                assert 'cpu' in str(
                    device), (f"lettuce is designed to work on cpu or cuda "
                              f"devices. {device} is not supported!")
                assert not use_native, ('can not use explicitly requested '
                                        'native extension on explicitly '
                                        'requested cpu device!')

        dtype = dtype or torch.float64  # default dtype to single
        assert dtype in [torch.float16, torch.float32, torch.float64], \
            (f"lettuce is designed to work with common float types "
             f"(16, 32 and 64 bit). {dtype.__name__} is not supported!")

        # store context configuration

        self.device = torch.device(device)
        self.dtype = dtype
        self.use_native = use_native

    def empty_tensor(self, size: Union[List[int], torch.Size], *args,
                     dtype=None, **kwargs) -> torch.Tensor:
        return torch.empty(size, *args, **kwargs, device=self.device,
                           dtype=(dtype or self.dtype))

    def zero_tensor(self, size: Union[List[int], torch.Size], *args,
                    dtype=None, **kwargs) -> torch.Tensor:
        return torch.zeros(size, *args, **kwargs, device=self.device,
                           dtype=(dtype or self.dtype))

    def one_tensor(self, size: Union[List[int], torch.Size], *args, dtype=None,
                   **kwargs) -> torch.Tensor:
        return torch.ones(size, *args, **kwargs, device=self.device,
                          dtype=(dtype or self.dtype))

    def convert_to_tensor(self, array, *args,
                          dtype: Optional[torch.dtype] = None,
                          **kwargs) -> torch.Tensor:
        if dtype is not None:
            return torch.tensor(array, *args, **kwargs, device=self.device,
                                dtype=dtype)

        is_bool_tensor = (isinstance(array, torch.Tensor)
                          and array.dtype in [bool, torch.uint8])
        is_bool_array = (isinstance(array, np.ndarray)
                         and array.dtype in [bool, np.uint8])

        if is_bool_tensor or is_bool_array:
            return torch.tensor(array, *args, **kwargs, device=self.device,
                                dtype=torch.uint8)
        else:
            if isinstance(array, torch.Tensor):
                return array.to(*args, **kwargs, device=self.device,
                                dtype=dtype)
            else:
                return torch.tensor(array, *args, **kwargs,
                                    device=self.device, dtype=self.dtype)

    @staticmethod
    def convert_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()
