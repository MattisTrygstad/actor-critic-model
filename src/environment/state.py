

from math import cos, pi, sin
from enums import BoardType, NodeState
from utils.config_parser import Config


class State:

    def __init__(self) -> None:
        super().__init__()

        self.neighbors = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
        self.size = Config.board_size

        self.nodes = {}  # (row, col): NodeStatus
        self.edges = []  # [((x,y),(i,j)),...)]
        self.node_names = {}  # (row, col): str
        self.node_coordinates = {}  # (row,col): (x_value, y_value)

        self.__initialize_board()

    def __initialize_board(self) -> None:
        # Generate board layout
        if Config.board_type == BoardType.TRIANGLE.value:
            for row in range(self.size):
                for col in range(row + 1):
                    self.nodes[(row, col)] = NodeState.OCCUPIED.value
                    self.node_names[(row, col)] = f'{row},{col}'
        elif Config.board_type == BoardType.DIAMOND.value:
            for row in range(self.size):
                for col in range(self.size):
                    # Constuct two triangles, which together will form a diamond pattern
                    if row >= col:
                        # Top triangle
                        self.nodes[(row, col)] = NodeState.OCCUPIED.value
                        self.node_names[(row, col)] = f'{row},{col}'
                    else:
                        # Bottom triangle
                        self.nodes[(row + self.size, col)] = NodeState.OCCUPIED.value
                        self.node_names[(row + self.size, col)] = f'{row + self.size}, {col}'

        # Set empty nodes
        for (row, col) in Config.empty_nodes:
            self.nodes[(row, col)] = NodeState.EMPTY.value

        # Add edges
        for (row, col), state in self.nodes.items():
            for (x, y) in self.neighbors:
                if (row + x, col + y) in self.nodes:
                    self.edges.append(((row, col), (row + x, col + y)))

        # Generate node coordinates used for visialization
        for (row, col) in self.nodes:
            # Rotate 90deg to match action offsets
            (x, y) = self.__rotation_matrix(row, col, -pi / 2)
            # Center each row to form a pyramid/diamond pattern (parallelogram -> diamond and right triangle -> equilateral triangle)
            self.node_coordinates[(row, col)] = (x + 1 / 2 * y, y)

    def get_empty_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.EMPTY.value}

    def get_occupied_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.OCCUPIED.value}

    def get_legal_actions(self) -> list:
        for (row, col), state in self.get_occupied_nodes().items():
            for x, y in self.neighbors:
                jumped_node = (row + x, col + y)
                landing_node = (row + 2 * x, col + 2 * y)

                # Check if jumped node and landing node is on the grid
                if jumped_node not in self.nodes or landing_node not in self.nodes:
                    continue

                # Check if Node state implies valid action
                if self.nodes[jumped_node] == NodeState.OCCUPIED.value and self.nodes[landing_node] == NodeState.EMPTY.value:
                    print((row, col))

    def __rotation_matrix(self, x: int, y: int, rad: float) -> tuple:
        return (x * cos(rad) - y * sin(rad), x * sin(rad) + y * cos(rad))
