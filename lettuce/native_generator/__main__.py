from . import *
from .. import all_stencils

all_streamings = [NativeStreamingStandard]
all_equilibriums = [NativeEquilibriumQuadratic]
all_collisions = [NativeCollisionBGK]


def main():
    all_mat = GeneratorModule.Matrix(all_stencils, all_streamings, all_equilibriums, all_collisions)

    # combine all equilibrium & collision with no stream:
    # > test_equilibrium_and_collision_mat = ModuleMatrix(
    # >     [D2Q9],
    # >     [NoStream],
    # >     all_equilibrium,
    # >     all_collision,
    # >     support_no_stream=[False],
    # >     support_no_collision=[False])

    # combine all stream with no collision:
    # > test_stream_mat = ModuleMatrix(
    # >     [D2Q9],
    # >     all_stream,
    # >     [QuadraticEquilibrium],
    # >     [NoCollision],
    # >     support_no_stream=[False],
    # >     support_no_collision=[False])

    GeneratorModule([all_mat], pretty_print=True).create_module()


if __name__ == '__main__':
    main()
