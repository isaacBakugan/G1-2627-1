import math

# Function b: z = 20 + x - 10cos(2*pi*x) + y - 10cos(2*pi*y), with x, y in [-10, 10]
BOUNDS = (-10, 10)
VARIABLE_COUNT = 2


def fitness(*variables):
    x, y = variables[0], variables[1]
    return 20 + x - 10 * math.cos(2 * math.pi * x) + y - 10 * math.cos(2 * math.pi * y)
