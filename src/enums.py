
from enum import Enum


class NodeState(Enum):
    EMPTY = 0
    OCCUPIED = 1


class Color(Enum):
    WHITE = '#F1FAEE'
    DARK_BLUE = '#1D3557'
    LIGHT_BLUE = '#457B9D'
    RED = '#E63946'


class BoardType(Enum):
    TRIANGLE = 0
    DIAMOND = 1
