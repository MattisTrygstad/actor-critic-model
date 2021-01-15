

from math import cos, pi, sin
from matplotlib import pyplot as plt
import numpy as np
from enums import BoardType, NodeState
from environment.grid_node import GridNode
from utils.config_parser import Config
import networkx as nx


class State:

    def __init__(self) -> None:
        super().__init__()

        self.size = Config.board_size

        self.nodes = {}  # (row, col): NodeStatus
        self.node_names = {}  # (row, col): str
        self.node_positions = {}  # (row,col): (x_value, y_value)

        # Generate board layout
        if Config.board_type == BoardType.TRIANGLE.value:
            for row in range(self.size):
                for col in range(row + 1):
                    self.nodes[(row, col)] = NodeState.OCCUPIED.value
                    self.node_names[(row, col)] = f'{row},{col}'
        elif Config.board_type == BoardType.DIAMOND.value:
            for row in range(self.size):
                for col in range(self.size):
                    self.nodes[(row, col)] = NodeState.OCCUPIED.value
                    self.node_names[(row, col)] = f'{row},{col}'

        # Set empty nodes
        for (row, col) in Config.empty_nodes:
            self.nodes[(row, col)] = NodeState.EMPTY.value

        # Generate positions used for visialization (i.e square -> diamond pattern)
        if Config.board_type == BoardType.TRIANGLE.value:
            for (row, col) in self.nodes:
                self.node_positions[(row, col)] = (-row + 0.5 * col, col)
        elif Config.board_type == BoardType.DIAMOND.value:
            for (row, col) in self.nodes:
                self.node_positions[(row, col)] = (row * cos(-pi / 4) - col * sin(-pi / 4), (row * sin(-pi / 4) + col * cos(-pi / 4)))

    def get_empty_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.EMPTY.value}

    def get_occupied_nodes(self) -> dict:
        return {key: value for (key, value) in self.nodes.items() if value == NodeState.OCCUPIED.value}

    def visualize(self) -> None:
        nodes = self.nodes
        node_names = self.node_names
        node_positions = self.node_positions

        fig = plt.figure(figsize=(9, 7))
        plt.axes()

        G = nx.Graph()

        nx.draw(G, pos=node_positions, nodelist=self.get_empty_nodes(), node_color='grey', node_size=800, ax=fig.axes[0], labels=node_names)
        nx.draw(G, pos=node_positions, nodelist=self.get_occupied_nodes(), node_color='blue', node_size=800, ax=fig.axes[0], labels=node_names)

        plt.show()
