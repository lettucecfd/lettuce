from lettuce.gencuda import *


def main():
    """
    """

    mat = ModuleMatrix([D1Q3(), D2Q9(), D3Q19(), D3Q27()],
                       [StandardStream()],
                       [QuadraticEquilibrium()],
                       [BKGCollision()])

    ModuleGenerator([mat], pretty_print=True).create_module()


if __name__ == '__main__':
    main()
