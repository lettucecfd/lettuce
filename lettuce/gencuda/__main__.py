from lettuce.gencuda import *

all_stencil = [D1Q3(), D2Q9(), D3Q19(), D3Q27()]
all_stream = [StandardStream()]
all_equilibrium = [QuadraticEquilibrium()]
all_collision = [BGKCollision()]


def main():
    """
    """

    all_mat = ModuleMatrix(all_stencil, all_stream, all_equilibrium, all_collision)

    # combine all equilibrium & collision with no stream:
    # > test_equilibrium_and_collision_mat = ModuleMatrix(
    # >     [D2Q9()],
    # >     [NoStream()],
    # >     all_equilibrium,
    # >     all_collision,
    # >     support_no_stream=[False],
    # >     support_no_collision=[False])

    # combine all stream with no collision:
    # > test_stream_mat = ModuleMatrix(
    # >     [D2Q9()],
    # >     all_stream,
    # >     [QuadraticEquilibrium()],
    # >     [NoCollision()],
    # >     support_no_stream=[False],
    # >     support_no_collision=[False])

    ModuleGenerator([all_mat], pretty_print=True).create_module()


if __name__ == '__main__':
    main()
