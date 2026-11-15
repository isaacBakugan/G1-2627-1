import math

# Function a: y = sin^2(x) / x, with x in [-10, 10]
BOUNDS = (-10, 10)
VARIABLE_COUNT = 1


def fitness(*variables):
    x = variables[0]
    # 0/0 is undefined, but the limit when x -> 0 is 0
    if x == 0:
        return 0.0
    return math.sin(x) ** 2 / x
