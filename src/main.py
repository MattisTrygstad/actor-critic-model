
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

    wins = 0
    losses = 0

    for episode in range(Config.episodes):
        critic.reset_eligibilies()
        actor.reset_eligibilities()

        state: UniversalState = env.get_state()
        legal_actions = env.get_legal_actions()

        action: UniversalAction = actor.generate_action(state, legal_actions)

        while True:
            reinforcement = env.execute_action(action)
            next_state: UniversalState = env.get_state()
            next_legal_actions = env.get_legal_actions()

            if env.check_win_condition():
                print('You won!')
                wins += 1
                break

            if len(next_legal_actions) == 0:
                print('You lost..')
                losses += 1
                break

            next_action: UniversalAction = actor.generate_action(next_state, next_legal_actions)
            print(next_action)
            print(str(next_state))

            actor.initialize_eligibility(state, action)

            td_error: float = critic.compute_temporal_difference_error(state, next_state, reinforcement)

            critic.initialize_eligibility(state)

            for legal_action in legal_actions:
                critic.approximator.compute_state_value(state, td_error, critic.eligibilities)
                critic.update_eligibility(state)

                actor.compute_policy(state, legal_action, td_error)
                actor.update_eligibility(state, legal_action)

            state = next_state
            action = next_action
            legal_actions = next_legal_actions

            # env.visualize(False)
            # plt.pause(0.01)
            #user_input = input('Enter any key to continue, q to quit: ')
            # if user_input == 'q':
            #    break

        env.reset()

    print(f'wins: {wins}, losses: {losses}')
    plt.close()


def normal_game():
    env = HexagonalGrid()

    env.visualize(False)

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
            env.undo_action()
            env.visualize(False)
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
        env.visualize(False)

    plt.close()


if __name__ == "__main__":
    main()
