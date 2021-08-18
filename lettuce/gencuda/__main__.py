from lettuce.gencuda import *


def main():
    """
    """

    # basic matrix
    basic = ModuleMatrix([D2Q9()],  # [D1Q3(), D2Q9(), D3Q19(), D3Q27()],
                         [StandardStream()],
                         [QuadraticEquilibrium()],
                         [BGKCollision()],
                         support_no_stream=[False],
                         support_no_collision=[False])

    # extra matrix
    collision_test = ModuleMatrix([D2Q9()],
                                  [NoStream()],
                                  [QuadraticEquilibrium()],
                                  [BGKCollision()],
                                  support_no_stream=[False],
                                  support_no_collision=[False])

    stream_test = ModuleMatrix([D2Q9()],
                               [StandardStream()],
                               [QuadraticEquilibrium()],
                               [NoCollision()],
                               support_no_stream=[False],
                               support_no_collision=[False])

    ModuleGenerator([basic, collision_test, stream_test], pretty_print=True).create_module()


if __name__ == '__main__':
    main()
