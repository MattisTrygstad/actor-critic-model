
from agent.learner import Learner
from environment.hexagonal_grid import HexagonalGrid
from utils.config_parser import Config


def main():
    grid = HexagonalGrid()
    agent = Learner()

    legal_action = grid.get_legal_actions()
    print(legal_action)
    grid.visualize()


if __name__ == "__main__":
    main()
