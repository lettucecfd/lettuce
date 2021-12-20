import json
from typing import Type

from . import *


class NativeStencil(NativeLatticeBase):
    _name: str
    _stencil: Type['Stencil']

    def __init__(self, stencil: Type['Stencil']):
        self._name = stencil.__name__.lower()
        self.stencil = stencil

    @property
    def name(self):
        return self._name

    def generate_d(self, generator: 'Generator'):
        if not generator.registered('d'):
            generator.register('d')

            # generate
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
            buffer = [f"{{{', '.join([str(x) + '.0' for x in e])}}}" for e in self.stencil.e]
            generator.append_constexpr_buffer(f"constexpr scalar_t e[q][d] = {{{', '.join(buffer)}}};")

    def generate_w(self, generator: 'Generator'):
        if not generator.registered('w'):
            generator.register('w')

            # requirements
            self.generate_q(generator)

            # generate
            buffer = [json.dumps(w) for w in self.stencil.w]
            buffer = ',\n                               '.join(buffer)
            generator.append_constexpr_buffer(f"constexpr scalar_t w[q] = {{{buffer}}};")

    def generate_cs(self, generator: 'Generator'):
        if not generator.registered('cs'):
            generator.register('cs')

            # generate
            generator.append_constexpr_buffer(f"constexpr scalar_t cs = {json.dumps(self.stencil.cs)};")
