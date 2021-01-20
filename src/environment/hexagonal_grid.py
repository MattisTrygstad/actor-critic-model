
from copy import deepcopy
from matplotlib import pyplot as plt
from abstract_classes.environment import Environment
from enums import BoardType, Color, NodeState
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

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges)

    def execute_action(self, action: tuple) -> None:
        self.history.append(deepcopy(self.state.nodes))

        (start_node, end_node) = action
        jumped_node = ((start_node[0] + end_node[0]) / 2, (start_node[1] + end_node[1]) / 2)

        self.state.start_pos = start_node
        self.state.end_pos = end_node

        self.state.nodes[start_node] = NodeState.EMPTY.value
        self.state.nodes[jumped_node] = NodeState.EMPTY.value
        self.state.nodes[end_node] = NodeState.OCCUPIED.value

    def undo_action(self) -> State:
        if self.history:
            self.state.nodes = self.history.pop()

    def get_legal_actions(self) -> list:
        return self.state.get_legal_actions()

    def check_win_condition(self) -> bool:
        num_occupied_nodes = len(self.state.get_occupied_nodes())
        if num_occupied_nodes == 1:
            return True
        else:
            return False

    def get_state(self) -> State:
        return self.state

    def reset(self) -> State:
        self.state.reset()

    def visualize(self) -> None:
        plt.cla()
        empty_nodes = self.state.get_empty_nodes()
        occupied_nodes = self.state.get_occupied_nodes()
        node_names = self.state.node_names
        node_coordinates = self.state.node_coordinates

        nx.draw(self.G, pos=node_coordinates, nodelist=empty_nodes, node_color=Color.LIGHT_BLUE.value, node_size=800)
        nx.draw(self.G, pos=node_coordinates, nodelist=occupied_nodes, node_color=Color.DARK_BLUE.value, node_size=800, ax=self.ax, labels=node_names, font_color=Color.WHITE.value)

        if self.history:
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.start_pos], node_color=Color.RED.value, node_size=1200)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.start_pos], node_color=Color.LIGHT_BLUE.value, node_size=800)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.end_pos], node_color=Color.RED.value, node_size=1200)
            nx.draw(self.G, pos=node_coordinates, nodelist=[self.state.end_pos], node_color=Color.DARK_BLUE.value, node_size=800)

        """ self.ax.set_axis_on()
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.title('Peg Solitaire') """

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.patch.set_facecolor(Color.WHITE.value)
        plt.pause(0.1)
