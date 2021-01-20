
import ast

from matplotlib import pyplot as plt
from agent.actor import Actor
from agent.critic import Critic
from agent.table_approximator import TableApproximator
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config


def main():
    if Config.human_mode:
        normal_game()
        return
    else:
        actor_critic_game()


def actor_critic_game():
    env = HexagonalGrid()

    critic = Critic(TableApproximator())
    actor = Actor()

    for episode in range(Config.episodes):
        critic.reset_eligibilies()
        actor.reset_eligibilities()

        prev_state: UniversalState = env.get_state()
        prev_action: UniversalAction = actor.generate_action(prev_state)

        goal = env.check_win_condition()
        legal_actions = env.get_legal_actions()
        while not goal and len(legal_actions) != 0:
            reinforcement = env.execute_action(prev_action)

            state: UniversalState = env.get_state()
            action: UniversalAction = actor.generate_action(state)

            actor.initialize_eligibility(prev_state, prev_action)

            td_error: float = critic.compute_temporal_difference_error(prev_state, state, reinforcement)

            critic.initialize_eligibility(prev_state)

            for action in legal_actions:
                critic.approximator.compute_state_value(prev_state, td_error, critic.eligibilities)
                critic.update_eligibility(prev_state)

                # Continue here with line 6c in algo


def normal_game():
    env = HexagonalGrid()

    env.visualize()

    while True:
        # Check win condition
        if env.check_win_condition():
            print('Congratulations, you won!')
            break

        legal_actions = env.get_legal_actions()

        print('-----\nLegal moves:')
        for action in legal_actions:
            print(f'From: {action[0]}, To: {action[1]}')
        print('-----')

        first_input = input('Enter start node: ')
        if first_input == 'q':
            break

        if first_input == 'undo':
            print('Action reversed')
            print(env.state.get_empty_nodes())
            env.undo_action()
            print(env.state.get_empty_nodes())
            env.visualize()
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

        env.execute_action((start_node, end_node))
        env.visualize()

    plt.close()


if __name__ == "__main__":
    main()
