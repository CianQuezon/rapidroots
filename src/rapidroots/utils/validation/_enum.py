from enum import IntEnum

class BracketCode(IntEnum):
    VALID = 0
    ORDER = 1       # a >= b
    NONFINITE = 2   # Nan/inf in a, b, fa, fb
    NOSIGN = 3      # no sign change and no endpoint ~0