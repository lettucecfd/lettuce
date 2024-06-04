from ... import NativeBoundary

__all__ = ['NativeBounceBackBoundary']


class NativeBounceBackBoundary(NativeBoundary):

    def __init__(self, index):
        NativeBoundary.__init__(self, index)

    @staticmethod
    def create(index):
        return NativeBounceBackBoundary(index)

    def generate(self, generator: 'Generator'):
        generator.append_pipeline_buffer(f"if (no_collision_mask[node_index] == {self.index})")
        generator.append_pipeline_buffer('{                      ')
        generator.append_pipeline_buffer('  scalar_t bounce[q];  ')

        for i in range(generator.stencil.q):
            generator.append_pipeline_buffer(f"  bounce[{i}] = f_reg[{generator.stencil.opposite[i]}];")

        for i in range(generator.stencil.q):
            generator.append_pipeline_buffer(f"  f_reg[{i}] = bounce[{i}];")

        generator.append_pipeline_buffer('}')
