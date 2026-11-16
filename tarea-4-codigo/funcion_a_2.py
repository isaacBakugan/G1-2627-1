
import math

BOUNDS = (-10, 10)
VARIABLE_COUNT = 1


def fitness(*variables):
    x = variables[0]

    if x == 0:
        return 0.0

    return (math.sin(x) ** 2) / x
