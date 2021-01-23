
import ast
from datetime import timedelta
from statistics import mean, median
import time

from matplotlib import pyplot as plt
from agent.actor import Actor
from agent.critic import Critic
from agent.table_approximator import TableApproximator
from environment.hexagonal_grid import HexagonalGrid
from environment.universal_action import UniversalAction
from environment.universal_state import UniversalState
from utils.config_parser import Config
from utils.normal_game import normal_game


def main():
    if Config.human_mode:
        normal_game()

    elif Config.experiments:
        run_experiments()

    else:
        actor_critic_game(Config.actor_learning_rate, Config.critic_learning_rate, Config.actor_decay_rate, Config.critic_decay_rate, Config.actor_discount_factor, Config.critic_discount_factor, Config.linear_epsilon, True)


def run_experiments() -> None:
    actor_learnig_rates = Config.actor_learning_rates
    critic_learnig_rates = Config.critic_learning_rates
    iterations = Config.iterations
    f = open('../experiment_results.txt', 'a')
    f.write('--- New Experiment ---\n')

    total = len(actor_learnig_rates) * len(critic_learnig_rates) * iterations * 2
    count = 1

    for actor_learning_rate in actor_learnig_rates:
        for critic_learning_rate in critic_learnig_rates:
            for x in range(2):

                if x == 0:
                    linear_epsilon = True
                else:
                    linear_epsilon = False

                training_wins = []
                test_wins = []

                estimated_run_time = 0

                for x in range(iterations):
                    start = time.time()
                    print(f'Experiment progress: {count}/{total}, estimated run time: {str(timedelta(seconds=estimated_run_time))}')
                    training, test = actor_critic_game(actor_learning_rate, critic_learning_rate, Config.actor_decay_rate, Config.critic_decay_rate, Config.actor_discount_factor, Config.critic_discount_factor, linear_epsilon, False)

                    training_wins.append(training)
                    test_wins.append(test)

                    count += 1

                    end = time.time()
                    estimated_run_time = (end - start) * (total - count)

                f.write(f'{actor_learning_rate},{critic_learning_rate},{linear_epsilon},{round(mean(training_wins), 2)},{round(median(training_wins),2)},{round(mean(test_wins), 2)},{round(median(test_wins),2)}\n')

    f.close()


def actor_critic_game(actor_learning_rate: float, critic_learning_rate: float, actor_decay_rate: float, critic_decay_rate: float, actor_discount_factor: float, critic_discount_factor: float, linear_epsilon: bool, visualize: bool) -> int:

    env = HexagonalGrid()
    approximator = TableApproximator()
    critic = Critic(approximator, critic_discount_factor, critic_decay_rate, critic_learning_rate)
    actor = Actor(actor_discount_factor, actor_decay_rate, actor_learning_rate)

    # Exploration vs. exploitation configuration
    initial_epsilon = Config.epsilon
    epsilon = initial_epsilon
    epsilon_linear = epsilon / Config.episodes

    # Statistics
    training_wins = 0
    test_wins = 0
    losses = 0
    remaining_nodes = []
    random_moves = []

    for episode in range(Config.episodes + Config.test_episodes):
        env.reset()

        if episode > Config.episodes:
            # No exploration during final model test
            epsilon = 0
        elif linear_epsilon:
            epsilon = initial_epsilon - epsilon_linear * episode
        else:
            epsilon *= Config.epsilon_decay

        state: UniversalState = env.get_state()
        action, random = actor.generate_action(state, env.get_legal_actions(), epsilon)

        history = []
        random_count = 0

        while True:
            reinforcement = env.execute_action(action)
            if random:
                random_count += 1

            if env.check_win_condition():
                if episode < Config.episodes:
                    training_wins += 1
                else:
                    test_wins += 1
                break

            if len(env.get_legal_actions()) == 0:
                losses += 1
                break

            next_state = env.get_state()
            next_legal_actions = env.get_legal_actions()
            next_action, random = actor.generate_action(next_state, next_legal_actions, epsilon)

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

        if visualize:
            random_moves.append(random_count)
            remaining_nodes.append(len(env.state.get_occupied_nodes()))

            if episode < Config.episodes:
                print(f'Episode: {episode}, wins: {training_wins}, losses: {losses}, epsilon: {round(epsilon, 5)}')
            if episode == Config.episodes:
                print(f'Testing final model...')

    plt.close()

    if visualize:
        print(f'Final model win rate: {test_wins}/{Config.test_episodes} = {round(test_wins/Config.test_episodes*100, 2)}% ')

        plt.plot(remaining_nodes)
        plt.ylabel('Remaining nodes')
        plt.show()

    return training_wins, test_wins


if __name__ == "__main__":
    main()
