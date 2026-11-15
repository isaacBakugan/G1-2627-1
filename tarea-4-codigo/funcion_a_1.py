"""Function a: y = sin(x)^2 / x, with x between -10 and 10."""

import math

BOUNDS = (-10, 10)  # limits of every variable
VARIABLE_COUNT = 1  # one variable: x


def fitness(*variables):
    """Return sin(x)^2 / x (higher is better)."""
    x = variables[0]
    if x == 0:
        # 0/0 is undefined, but the limit of sin(x)^2 / x when x -> 0 is 0
        return 0.0
    return math.sin(x) ** 2 / x
