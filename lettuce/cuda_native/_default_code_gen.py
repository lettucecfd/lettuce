import json
from enum import Enum
from functools import lru_cache as cached
from typing import List, Optional, cast

from . import *
from ._registry import Registry
from .. import Stencil

__all__ = ['StreamingStrategy', 'DefaultCodeGeneration']


class StreamingStrategy(Enum):
    NO_STREAMING = 0b00
    PRE_STREAMING = 0b10
    POST_STREAMING = 0b01
    DOUBLE_STREAMING = 0b11

    @cached
    def pre_streaming(self) -> bool:
        return bool(self.value & StreamingStrategy.PRE_STREAMING.value)

    @cached
    def post_streaming(self) -> bool:
        return bool(self.value & StreamingStrategy.POST_STREAMING.value)


class DefaultCodeGeneration(Registry):
    _xyz: str = "xyz"
    _cxyz: str = "cxyz"

    stencil: 'Stencil'
    transformer: List['NativeTransformer']
    equilibrium: Optional['NativeEquilibrium']
    streaming_strategy: StreamingStrategy
    _transformer_mask: bool

    def __init__(self, stencil: 'Stencil',
                 transformer: Optional[List['NativeTransformer']],
                 equilibrium: Optional['NativeEquilibrium'],
                 streaming_strategy: StreamingStrategy):

        Registry.__init__(self, stencil.d)
        self.stencil = stencil
        self.transformer = transformer
        self.equilibrium = equilibrium
        self.streaming_strategy = streaming_strategy

        self._transformer_mask = len(transformer) > 1

    @property
    def collision(self) -> NativeCollision:
        return cast(NativeCollision, self.transformer[0])

    @property
    def boundaries(self) -> List[NativeBoundary]:
        return cast(List[NativeBoundary], self.transformer[1:])

    @cached
    def d(self) -> str:
        return f"static_cast<index_t>({json.dumps(self.stencil.d)})"

    @cached
    def q(self) -> str:
        return f"static_cast<index_t>({json.dumps(self.stencil.q)})"

    @cached
    def e(self, q: int, d: int) -> str:
        assert d in range(self.stencil.d)
        assert q in range(self.stencil.q)
        return f"static_cast<index_t>({json.dumps(self.stencil.e[q][d])})"

    @cached
    def w(self, q: int) -> str:
        assert q in range(self.stencil.q)
        return f"static_cast<scalar_t>({json.dumps(self.stencil.w[q])})"

    @cached
    def opposite(self, q: int) -> str:
        assert q in range(self.stencil.q)
        return f"static_cast<index_t>({json.dumps(self.stencil.opposite[q])})"

    @cached
    def support_no_collision_mask(self) -> str:
        return str(self._transformer_mask).lower()

    @cached
    def support_no_streaming_mask(self) -> str:
        return str(self._transformer_mask).lower()

    @cached
    def thread_count(self, d: Optional[int] = None) -> str:
        if d is None:
            values = [self.thread_count(d) for d in range(self.stencil.d)]
            return self.cuda.variable('dim3', 'thread_count', f"dim3{{{','.join(values)}}}")

        assert d in range(self.stencil.d)
        value = 'static_cast<index_t>(16)' if self.stencil.d < 3 else 'static_cast<index_t>(8)'
        return value

    @cached
    def cuda_block_count(self, d: Optional[int] = None) -> str:
        if d is None:
            values = [f"{self.cuda_size(d + 1)}/{self.thread_count(d)}" for d in range(self.stencil.d)]
            return self.cuda.variable('dim3', 'block_count', f"dim3{{{','.join(values)}}}")

        assert d in range(self.stencil.d)
        return f"block_count.{'cxyz'[d]}"

    @cached
    def kernel_block_count(self, d: int) -> str:
        assert d in range(self.stencil.d)
        variable = self.cuda_block_count(d)
        return self.kernel_hook(variable, Parameter('index_t', variable.replace('.', '_')))

    @cached
    def cuda_size(self, d: int) -> str:
        assert d in range(self.stencil.d + 1)
        if d == 0:
            return f"{self.stencil.q}"
        return f"static_cast<index_t>({self.cuda_f()}.size({d}))"

    @cached
    def kernel_size(self, d: int) -> str:
        if d == 0:
            return f"{self.stencil.q}"
        return self.kernel_hook(self.cuda_size(d), Parameter('index_t', f"size_{'cxyz'[d]}"))

    @cached
    def kernel_index(self, d: int) -> str:
        assert d in range(self.stencil.d)

        variable = 'xyz'[d]
        expr = f"blockIdx.{'xyz'[d]}*blockDim.{'xyz'[d]}+threadIdx.{'xyz'[d]}"
        return self.pipes.variable('index_t', variable, expr)

    @cached
    def cuda_stride(self, d: int) -> str:
        assert d in range(self.stencil.d + 1)
        return f"static_cast<index_t>({self.cuda_f()}.stride({d}))"

    @cached
    def kernel_stride(self, d: int) -> str:
        return self.kernel_hook(self.cuda_stride(d), Parameter('index_t', f"stride_{'cxyz'[d]}"))

    @cached
    def kernel_offset(self, d: int, v: int) -> str:
        assert d in range(self.stencil.d)
        assert v in [-1, 0, 1]

        if v == 0:
            variable = f"offset_{'xyz'[d]}"
            code = f"{self.kernel_index(d)}*{self.kernel_stride(d + 1)}"

        elif v == 1:
            variable = f"offset_{'xyz'[d]}_h"
            # todo benchmark branch less optimization
            code = f"({self.kernel_index(d)}!={self.kernel_size(d + 1)}-1)*({self.kernel_index(d)}+1)*{self.kernel_stride(d + 1)}"
            # code = f"(({self.kernel_index(d)}=={self.kernel_size(d + 1)}-1)?0:{self.kernel_index(d)}+1)*{self.kernel_stride(d + 1)}"

        else:  # v == -1:
            variable = f"offset_{'xyz'[d]}_l"
            # todo benchmark branch less optimization
            code = f"(({self.kernel_index(d)}-1)+(({self.kernel_size(d + 1)}-{self.kernel_index(d)})&(-({self.kernel_index(d)}==0))))*{self.kernel_stride(d + 1)}"
            # code = f"(({self.kernel_index(d)}==0)?{self.kernel_size(d + 1)}-1:{self.kernel_index(d)}-1)*{self.kernel_stride(d + 1)}"

        return self.pipes.variable('index_t', variable, code)

    @cached
    def kernel_base_index(self) -> str:
        expr = '+'.join(self.kernel_offset(d, 0) for d in range(self.stencil.d))
        return self.pipes.variable('index_t', 'base_index', expr)

    def assert_ncm(self):
        assert self._transformer_mask
        code = "assert hasattr(simulation, 'no_collision_mask')"
        self.python_pre.append(code, guard=code)

    def assert_nsm(self):
        assert self._transformer_mask
        code = "assert hasattr(simulation, 'no_streaming_mask')"
        self.python_pre.append(code, guard=code)

    @cached
    def cuda_no_collision_mask(self) -> str:
        self.assert_ncm()
        return self.cuda_hook('simulation.no_collision_mask', Parameter('at::Tensor', 'ncm'))

    def cuda_ncm(self) -> str:
        return self.cuda_no_collision_mask()

    @cached
    def kernel_no_collision_mask(self) -> str:
        variable_ncm = self.cuda_no_collision_mask()
        return self.kernel_hook(f"{variable_ncm}.data_ptr<byte_t>()", Parameter('byte_t*', variable_ncm))

    def kernel_ncm(self) -> str:
        return self.kernel_no_collision_mask()

    @cached
    def cuda_no_streaming_mask(self) -> str:
        self.assert_nsm()
        return self.cuda_hook('simulation.no_streaming_mask', Parameter('at::Tensor', 'nsm'))

    def cuda_nsm(self) -> str:
        return self.cuda_no_streaming_mask()

    @cached
    def kernel_no_streaming_mask(self) -> str:
        self.assert_nsm()
        variable_nsm = self.cuda_no_streaming_mask()
        return self.kernel_hook(f"{variable_nsm}.data_ptr<byte_t>()", Parameter('byte_t*', variable_nsm))

    def kernel_nsm(self) -> str:
        return self.kernel_no_streaming_mask()

    @cached
    def cuda_f(self) -> str:
        return self.cuda_hook(f"simulation.flow.f", Parameter('at::Tensor', 'f'))

    @cached
    def kernel_f(self) -> str:
        variable = self.cuda_f()
        return self.kernel_hook(f"{variable}.data_ptr<scalar_t>()", Parameter('scalar_t*', variable))

    @cached
    def kernel_stream_offset(self, q: int) -> str:
        assert q in range(self.stencil.q)
        offsets = [self.kernel_offset(d, self.stencil.e[q][d]) for d in range(self.stencil.d)]
        return self.pipes.variable('index_t', f"stream_offset_{q}", '+'.join(offsets))

    @cached
    def f_reg(self, q: int) -> str:
        assert q in range(self.stencil.q)

        variable = f"f_reg_{q}"

        f = self.kernel_f()
        stride_c = self.kernel_stride(0)

        if not self.streaming_strategy.pre_streaming():  # no pre streaming
            base = self.kernel_base_index()
            return self.pipes.mutable('scalar_t', variable, f"{f}[{q}*{stride_c}+{base}]")

        # pre streaming
        code = f"{f}[{q}*{stride_c}+{self.kernel_stream_offset(self.stencil.opposite[q])}]"

        if self._transformer_mask:
            base = self.kernel_base_index()
            # prepend optional no streaming
            code = f"{self.kernel_nsm()}[{q}*{stride_c}+{base}]?{f}[{q}*{stride_c}+{base}]:{code}"

        # pre streaming
        return self.pipes.mutable('scalar_t', variable, code)

    @cached
    def kernel_rho(self) -> str:
        return self.pipes.variable('scalar_t', 'rho', '+'.join(self.f_reg(q) for q in range(self.stencil.q)))

    @cached
    def kernel_rho_inv(self) -> str:
        return self.pipes.variable('scalar_t', 'rho_inv', f"1.0/{self.kernel_rho()}")

    @cached
    def kernel_u(self, d: int) -> str:
        assert d in range(self.stencil.d)

        exf = '+'.join(f"{self.e(q, d)}*{self.f_reg(q)}" for q in range(self.stencil.q))
        if self.stencil.d > 1:
            expr = f"({exf})*{self.kernel_rho_inv()}"
        else:
            expr = f"({exf})/{self.kernel_rho()}"

        return self.pipes.variable('scalar_t', f"u_{'xyz'[d]}", expr)

    @cached
    def cuda_f_next(self) -> str:
        return self.cuda_hook('simulation.flow.f_next', Parameter('at::Tensor', 'f_next'))

    @cached
    def kernel_f_next(self) -> str:
        variable = self.cuda_f_next()
        return self.kernel_hook(f"{variable}.data_ptr<scalar_t>()", Parameter('scalar_t*', variable))

    def generate(self):
        self.thread_count()
        self.cuda_block_count()

        for transformer in self.transformer:
            if self._transformer_mask:
                self.pipe.append(f"if({self.kernel_ncm()}[{self.kernel_base_index()}]=={transformer.index}){{")
                transformer.generate(self)
                self.pipe.append('}')
            else:
                transformer.generate(self)

        f_next = self.kernel_f_next()
        for q in range(self.stencil.q):
            stride_c = self.kernel_stride(0)

            if not self.streaming_strategy.post_streaming():
                base = self.kernel_base_index()
                self.pipe.append(f"{f_next}[{q}*{stride_c}+{base}]={self.f_reg(q)};")
                continue

            if self._transformer_mask:
                base = self.kernel_base_index()
                self.pipe.append(f"if({self.kernel_no_streaming_mask()}[{q}*{stride_c}+{base}])")
                self.pipe.append(f"  {f_next}[{q}*{stride_c}+{base}]={self.f_reg(q)};")
                self.pipe.append('else')

            # post streaming
            self.pipe.append(f"{f_next}[{q}*{stride_c}+{self.kernel_stream_offset(q)}]={self.f_reg(q)};")
