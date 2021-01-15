
from enum import Enum


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    RIGHT_UP = 5
    LEFT_DOWN = 6


class NodeState(Enum):
    EMPTY = 0
    OCCUPIED = 1


class BoardType(Enum):
    TRIANGLE = 0
    DIAMOND = 1
