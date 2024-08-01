#ifndef {guard}
#define {guard}

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( push )
#pragma warning ( disable : 4067 )
#pragma warning ( disable : 4624 )
#endif

#include <torch/extension.h>

#if _MSC_VER && !__INTEL_COMPILER
#pragma warning ( pop )
#endif

void
lettuce_cuda_{name}({cuda_wrapper_parameter});

void
lettuce_{name}({cpp_wrapper_parameter});

#endif //{guard}
