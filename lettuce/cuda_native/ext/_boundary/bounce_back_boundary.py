from ... import *

__all__ = ['NativeBounceBackBoundary']


class NativeBounceBackBoundary(NativeBoundary):

    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeBounceBackBoundary(index)

    def generate(self, reg: 'DefaultCodeGeneration'):
        reg.pipe.append('{')
        reg.pipe.append(f"scalar_t bounce[{reg.stencil.q}];")
        for q in range(reg.stencil.q):
            reg.pipe.append(f"  bounce[{q}] = {reg.f_reg(reg.stencil.opposite[q])};")
        for q in range(reg.stencil.q):
            reg.pipe.append(f"  {reg.f_reg(q)} = bounce[{q}];")
        reg.pipe.append('}')
