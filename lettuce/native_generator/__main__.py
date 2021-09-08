from . import *
from . import _load_template
from .. import all_stencils

all_stencils = all_stencils()
all_streamings = [NativeStandardStreaming]
all_equilibriums = [NativeQuadraticEquilibrium]
all_collisions = [NativeBGKCollision]


def main():
    all_mats = [ModuleGenerator.Matrix(all_stencils, all_streamings, all_equilibriums, all_collisions)]

    # all_mats += [ModuleGenerator.Matrix(
    #     all_stencils,
    #     [NativeNoStreaming],
    #     all_equilibriums,
    #     all_collisions,
    #     support_no_stream=[False],
    #     support_no_collision=[False])]

    # all_mats += [ModuleGenerator.Matrix(
    #     all_stencils,
    #     all_streamings,
    #     [None],
    #     [NativeNoCollision],
    #     support_no_stream=[False],
    #     support_no_collision=[False])]

    ModuleGenerator(all_mats, pretty_print=True).create_module()


if __name__ == '__main__':
    main()
