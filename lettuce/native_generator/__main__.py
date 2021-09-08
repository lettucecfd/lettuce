from . import *
from .. import all_stencils

all_stencils = all_stencils()
all_streamings = [NativeStreamingStandard]
all_equilibriums = [NativeEquilibriumQuadratic]
all_collisions = [NativeCollisionBGK]


def main():
    all_mats = [GeneratorModule.Matrix(all_stencils, all_streamings, all_equilibriums, all_collisions)]

    all_mats += [GeneratorModule.Matrix(
        all_stencils,
        [NativeStreamingNo],
        all_equilibriums,
        all_collisions,
        support_no_stream=[False],
        support_no_collision=[False])]

    all_mats += [GeneratorModule.Matrix(
        all_stencils,
        all_streamings,
        [None],
        [NativeCollisionNo],
        support_no_stream=[False],
        support_no_collision=[False])]

    GeneratorModule(all_mats, pretty_print=True).create_module()


if __name__ == '__main__':
    main()
