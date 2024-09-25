from ... import NativeBoundary

__all__ = ['NativeBounceBackBoundary']


class NativeBounceBackBoundary(NativeBoundary):

    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeBounceBackBoundary(index)

    def generate(self, generator: 'Generator'):
        buffer = generator.append_pipeline_buffer
        ncm = generator.support_no_collision_mask

        buffer(f"if (no_collision_mask[node_index] == {self.index})", cond=ncm)
        buffer('{                      ')
        buffer('  scalar_t bounce[q];  ')

        for i in range(generator.stencil.q):
            buffer(f"  bounce[{i}] = f_reg[{generator.stencil.opposite[i]}];")

        for i in range(generator.stencil.q):
            buffer(f"  f_reg[{i}] = bounce[{i}];")

        buffer('}')
