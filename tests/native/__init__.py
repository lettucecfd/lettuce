from tests.common import *

if not torch.cuda.is_available():
    pytest.skip(reason="CUDA is not available on this machine.",
                allow_module_level=True)
