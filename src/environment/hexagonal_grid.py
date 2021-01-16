
from matplotlib import pyplot as plt
from abstract_classes.environment import Environment
from enums import Action, BoardType
from environment.state import State
import networkx as nx

from utils.config_parser import Config


class HexagonalGrid(Environment):

    def __init__(self):
        super().__init__()

        self.state = State()

        self.G = nx.Graph()
        self.G.add_nodes_from(self.state.nodes)
        self.G.add_edges_from(self.state.edges)

    def execute_action(self, action: Action) -> State:
        pass

    def reverse_action() -> State:
        pass

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

        if Config.board_type == BoardType.DIAMOND.value:
            fig = plt.figure(figsize=(7, 8))
        else:
            fig = plt.figure(figsize=(9, 7))
        plt.axes()

        nx.draw(self.G, pos=node_coordinates, nodelist=empty_nodes, node_color='grey', node_size=800, ax=fig.axes[0], labels=node_names)
        nx.draw(self.G, pos=node_coordinates, nodelist=occupied_nodes, node_color='blue', node_size=800, ax=fig.axes[0], labels=node_names)

        # plt.gca().invert_xaxis()
        plt.show()
