
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

    # Exploration vs. exploitation configuration
    initial_epsilon = Config.epsilon
    epsilon = initial_epsilon
    epsilon_decay = epsilon / Config.episodes

    # Statistics
    wins = 0
    losses = 0
    remaining_nodes = []

    for episode in range(Config.episodes):
        critic.reset_eligibilies()
        actor.reset_eligibilities()

        state: UniversalState = env.get_state()
        legal_actions = env.get_legal_actions()

        actor.initialize_state_action_pairs(state, legal_actions)
        critic.set_eligibility(state, 1)
        critic.approximator.initialize_state_value(state)

        epsilon = initial_epsilon - epsilon_decay * episode
        action: UniversalAction = actor.generate_action(state, legal_actions, epsilon)

        while True:
            reinforcement = env.execute_action(action)
            next_state: UniversalState = env.get_state()
            next_legal_actions = env.get_legal_actions()

            actor.initialize_state_action_pairs(next_state, next_legal_actions)
            critic.approximator.initialize_state_value(next_state)

            if env.check_win_condition():
                wins += 1
                break

            if len(next_legal_actions) == 0:
                losses += 1
                break

            next_action: UniversalAction = actor.generate_action(next_state, next_legal_actions, epsilon)

            actor.set_eligibility(next_state, next_action)
            critic.set_eligibility(next_state, 1)

            td_error: float = critic.compute_temporal_difference_error(state, next_state, reinforcement)

            # for all (s,a) pairs
            critic.approximator.compute_state_values(td_error, critic.eligibilities)
            critic.decay_eligibilies()
            actor.compute_policies(td_error)
            actor.decay_eligibilities()

            state = next_state
            action = next_action
            legal_actions = next_legal_actions

            # env.visualize(False)
            # plt.pause(0.01)
            #user_input = input('Enter any key to continue, q to quit: ')
            # if user_input == 'q':
            #    break

        print(f'Episode: {episode}, wins: {wins}, losses: {losses}, epsilon: {round(epsilon, 5)}')
        remaining_nodes.append(len(env.state.get_occupied_nodes()))
        env.reset()

    plt.close()

    plt.plot(remaining_nodes)
    plt.ylabel('Remaining nodes')
    plt.show()


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

        action = UniversalAction()
        action.action = (start_node, end_node)
        env.execute_action(action)
        env.visualize(False)

    plt.close()


if __name__ == "__main__":
    main()
