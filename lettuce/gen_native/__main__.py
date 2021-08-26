from lettuce.gen_native import *

all_stencil = [NativeD1Q3, NativeD2Q9, NativeD3Q19, NativeD3Q27]
all_stream = [NativeStreamingStandard]
all_equilibrium = [NativeEquilibriumQuadratic]
all_collision = [NativeCollisionBGK]


def main():
    """
    """

    all_mat = GeneratorModule.Matrix(all_stencil, all_stream, all_equilibrium, all_collision)

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
