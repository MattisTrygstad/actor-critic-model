
from copy import copy, deepcopy
import copy
from matplotlib import pyplot as plt
from abstract_classes.environment import Environment
from enums import BoardType, NodeState
from environment.state import State
import networkx as nx

from utils.config_parser import Config


class HexagonalGrid(Environment):

    def __init__(self):
        self.state = State()
        self.history = []

        if Config.board_type == BoardType.DIAMOND.value:
            self.fig, self.ax = plt.subplots(figsize=(7, 8))
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 7))

        plt.title('Peg Solitaire')

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges)

    def execute_action(self, action: tuple) -> None:
        self.history.append(deepcopy(self.state.nodes))
        (start_node, end_node) = action
        jumped_node = ((start_node[0] + end_node[0]) / 2, (start_node[1] + end_node[1]) / 2)

        self.state.nodes[start_node] = NodeState.EMPTY.value
        self.state.nodes[jumped_node] = NodeState.EMPTY.value
        self.state.nodes[end_node] = NodeState.OCCUPIED.value

    def undo_action(self) -> State:
        if self.history:
            self.state.nodes = self.history.pop()

    def get_legal_actions(self) -> list:
        return self.state.get_legal_actions()

    def get_state(self) -> State:
        return self.state

    def reset(self) -> State:
        self.state.reset()

    def visualize(self) -> None:
        empty_nodes = self.state.get_empty_nodes()
        occupied_nodes = self.state.get_occupied_nodes()
        node_names = self.state.node_names
        node_coordinates = self.state.node_coordinates

        nx.draw(self.G, pos=node_coordinates, nodelist=empty_nodes, node_color='grey', node_size=800, ax=self.ax, labels=node_names)
        nx.draw(self.G, pos=node_coordinates, nodelist=occupied_nodes, node_color='blue', node_size=800, ax=self.ax, labels=node_names)

        self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
