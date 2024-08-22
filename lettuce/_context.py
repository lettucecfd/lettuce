import torch
import numpy as np

from typing import List, Optional, Union

__all__ = ['Context']


class Context:
    def __init__(self, device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None,
                 use_native: Optional[bool] = None):
        if device is str:
            assert 'cuda' in device or 'cpu' in device
        # check sanity of context configuration

        if device is None and use_native is None:
            device = torch.device('cuda:0'
                                  if torch.cuda.is_available() else 'cpu')
            use_native = True if torch.cuda.is_available() else False

        elif device is None:
            assert not use_native or torch.cuda.is_available(), \
                ('cuda_native extension explicitly requested but '
                 'cuda is not available!')
            device = torch.device('cuda:0')

        elif use_native is None:
            if 'cuda' in str(device):
                assert torch.cuda.is_available(), \
                    ('cuda device explicitly requested but '
                     'cuda is not available!')
                use_native = True
            else:
                assert 'cpu' in str(device), \
                    (f"lettuce is designed to work on cpu or cuda devices. "
                     f"{device} is not supported!")
                use_native = False

        else:
            if 'cuda' in str(device):
                assert torch.cuda.is_available(), \
                    ('cuda device explicitly requested '
                     'but cuda is not available!')
            else:
                assert 'cpu' in str(device), \
                    (f"lettuce is designed to work on cpu or cuda devices. "
                     f"{device} is not supported!")
                assert not use_native, \
                    ('can not use explicitly requested cuda_native extension '
                     'on explicitly requested cpu device!')

        dtype = dtype or torch.float32  # default dtype to single
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
                          dtype: Optional[torch.dtype] = None, **kwargs
                          ) -> torch.Tensor:
        is_tensor = isinstance(array, torch.Tensor)
        new_dtype = dtype
        if dtype is None:
            if hasattr(array, 'dtype'):
                if array.dtype in [bool, torch.bool]:
                    new_dtype = torch.bool
                elif array.dtype in [bool, torch.uint8, np.uint8]:
                    new_dtype = torch.uint8
                else:
                    new_dtype = self.dtype
            else:
                new_dtype = self.dtype

        if is_tensor:
            return array.to(*args, **kwargs, device=self.device,
                            dtype=new_dtype)
        else:
            return torch.tensor(array, *args, **kwargs, device=self.device,
                                dtype=new_dtype)

    @staticmethod
    def convert_to_ndarray(tensor: Union[torch.Tensor, List]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.array(tensor)
