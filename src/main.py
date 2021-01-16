
from agent.learner import Learner
from environment.hexagonal_grid import HexagonalGrid
from utils.config_parser import Config


def main():
    grid = HexagonalGrid()
    agent = Learner()

    grid.get_legal_actions()
    grid.visualize()


if __name__ == "__main__":
    main()
