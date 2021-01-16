
import ast
from cmath import e

from matplotlib import pyplot as plt
from agent.learner import Learner
from environment.hexagonal_grid import HexagonalGrid
from utils.config_parser import Config


def main():
    if Config.human_mode:
        normal_game()
        return


def normal_game():
    grid = HexagonalGrid()

    grid.visualize()

    while True:
        legal_actions = grid.get_legal_actions()

        print('-----\nLegal moves:')
        for action in legal_actions:
            print(f'From: {action[0]}, To: {action[1]}')
        print('-----')

        first_input = input('Enter start node: ')
        if first_input == 'q':
            break

        if first_input == 'undo':
            print('Action reversed')
            print(grid.state.get_empty_nodes())
            grid.undo_action()
            print(grid.state.get_empty_nodes())
            grid.visualize()
            continue

        try:
            start_node = tuple(ast.literal_eval(first_input))
            end_node = tuple(ast.literal_eval(input('Enter end node: ')))
        except:
            print('Invalid input, try again!')
            continue

        if (start_node, end_node) not in legal_actions:
            print('Illegal move, try again!')
            continue

        grid.execute_action((start_node, end_node))
        grid.visualize()

    plt.close()


if __name__ == "__main__":
    main()
