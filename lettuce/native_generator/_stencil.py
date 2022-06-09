import json
from typing import Type

from . import *


class NativeStencil(NativeLatticeBase):
    _stencil: Type['Stencil']

    def __init__(self, stencil: Type['Stencil']):
        self.stencil = stencil

    @property
    def name(self):
        return self.stencil.__name__.lower()

    def generate_d(self, generator: 'Generator'):
        if not generator.registered('d'):
            generator.register('d')

            # generate
            generator.append_launcher_buffer(f"constexpr index_t d = {self.stencil.D()};")
            generator.append_constexpr_buffer(f"constexpr index_t d = {self.stencil.D()};")

    def generate_q(self, generator: 'Generator'):
        if not generator.registered('q'):
            generator.register('q')

            # generate
            generator.append_constexpr_buffer(f"constexpr index_t q = {self.stencil.Q()};")

    def generate_e(self, generator: 'Generator'):
        if not generator.registered('e'):
            generator.register('e')

            # requirements
            self.generate_d(generator)
            self.generate_q(generator)

            # generate
            constexpr_buf = generator.append_constexpr_buffer

            constexpr_buf('constexpr index_t e[q][d] = {')
            for i in self.stencil.e:
                constexpr_buf(f"  {{{', '.join([str(j) for j in i])}}},")

            constexpr_buf('};')

    def generate_w(self, generator: 'Generator'):
        if not generator.registered('w'):
            generator.register('w')

            # requirements
            self.generate_q(generator)

            # generate
            constexpr_buf = generator.append_constexpr_buffer

            constexpr_buf('constexpr scalar_t w[q] = {')
            for w in self.stencil.w:
                constexpr_buf(f"  {json.dumps(w)},")
            constexpr_buf('};')

    def generate_cs(self, generator: 'Generator'):
        if not generator.registered('cs'):
            generator.register('cs')

            # generate
            generator.append_constexpr_buffer(f"constexpr scalar_t cs = {json.dumps(self.stencil.cs)};")
