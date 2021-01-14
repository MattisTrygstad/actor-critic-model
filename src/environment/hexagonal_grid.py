
from matplotlib import pyplot as plt
from abstract_classes.environment import Environment
from environment.grid_node import GridNode
from utils.config_parser import Config

import networkx as nx


class HexagonalGrid(Environment):

    def __init__(self):
        super().__init__()

        self.size = Config.size

        self.nodes = [[GridNode(True, False)] * self.size] * self.size

        G = nx.petersen_graph()
        plt.subplot(121)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.subplot(122)
        nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
        plt.show()

    def method(self):
        pass
