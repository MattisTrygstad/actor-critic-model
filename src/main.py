
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
        new_actor_critic_game()
        # actor_critic_game()


def new_actor_critic_game():
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
        env.reset()
        epsilon = initial_epsilon - epsilon_decay * episode

        state: UniversalState = env.get_state()
        action: UniversalAction = actor.generate_action(state, env.get_legal_actions(), epsilon)

        history = []

        while True:
            reinforcement = env.execute_action(action)

            if env.check_win_condition():
                wins += 1
                break

            if len(env.get_legal_actions()) == 0:
                losses += 1
                break

            next_state = env.get_state()
            next_legal_actions = env.get_legal_actions()
            next_action = actor.generate_action(next_state, next_legal_actions, epsilon)

            actor.set_eligibility(state, action, 1)

            td_error = critic.compute_temporal_difference_error(state, next_state, reinforcement)

            critic.set_eligibility(state, 1)

            history.append((state, action))

            for num, [s, a] in enumerate(history):

                critic.compute_state_value(s, td_error)

                critic.set_eligibility(s, Config.critic_discount_factor * Config.critic_decay_rate * critic.eligibilities[str(s)])

                actor.compute_policy(s, a, td_error)

                actor.set_eligibility(s, a, Config.actor_discount_factor * Config.actor_decay_rate * actor.eligibilities[str(s)][str(a)])

            state = next_state
            action = next_action

        remaining_nodes.append(len(env.state.get_occupied_nodes()))
        print(f'Episode: {episode}, wins: {wins}, losses: {losses}, epsilon: {round(epsilon, 5)}')

    plt.close()

    plt.plot(remaining_nodes)
    plt.ylabel('Remaining nodes')
    plt.show()


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
        critic.initialize_state_value(state)

        epsilon = initial_epsilon - epsilon_decay * episode
        action: UniversalAction = actor.generate_action(state, legal_actions, epsilon)

        #critic.set_eligibility(state, 1)
        #actor.set_eligibility(state, action, 1)

        while True:
            reinforcement = env.execute_action(action)
            next_state: UniversalState = env.get_state()
            next_legal_actions = env.get_legal_actions()

            actor.initialize_state_action_pairs(next_state, next_legal_actions)
            critic.initialize_state_value(next_state)

            if env.check_win_condition():
                wins += 1
                break

            if len(next_legal_actions) == 0:
                losses += 1
                break

            next_action: UniversalAction = actor.generate_action(next_state, next_legal_actions, epsilon)

            actor.set_eligibility(state, action, 1)

            td_error: float = critic.compute_temporal_difference_error(state, next_state, reinforcement)

            critic.set_eligibility(state, 1)

            # for all (s,a) pairs
            critic.compute_state_values(td_error)
            critic.decay_eligibilies()
            actor.compute_policies(td_error)
            actor.decay_eligibilities()

            state = next_state
            action = next_action
            legal_actions = next_legal_actions

            """ env.visualize(False)
            plt.pause(0.01)
            print('actor eli:', actor.eligibilities)
            print('critic eli:', critic.eligibilities)

            user_input = input('Enter any key to continue, q to quit: ')
            if user_input == 'q':
                break """

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
