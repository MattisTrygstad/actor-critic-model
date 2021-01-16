
from enum import Enum


class NodeState(Enum):
    EMPTY = 0
    OCCUPIED = 1


class BoardType(Enum):
    TRIANGLE = 0
    DIAMOND = 1
