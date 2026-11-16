
import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 2


def fitness(*variables):
    x, y = variables

    z = (
        20
        + x - 10 * math.cos(2 * math.pi * x)
        + y - 10 * math.cos(2 * math.pi * y)
    )

    return z
