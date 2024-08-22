from tests.conftest import *

if not torch.cuda.is_available():
    pytest.skip(reason="CUDA is not available on this machine, "
                       "so cuda_native will not be tested..",
                allow_module_level=True)
