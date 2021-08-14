from lettuce.gencuda import *


def main():
    """
    """

    # stencils = [D1Q3(), D2Q9(), D3Q19(), D3Q27()]
    stencils = [D2Q9()]

    mat = ModuleMatrix(stencils,
                       [StandardStream()],
                       [QuadraticEquilibrium()],
                       [BKGCollision()],
                       support_no_stream=[False],
                       support_no_collision=[False])

    ModuleGenerator([mat], pretty_print=True).create_module()


if __name__ == '__main__':
    main()
