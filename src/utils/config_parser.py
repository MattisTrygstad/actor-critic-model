
import ast
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    human_mode = bool(ast.literal_eval(config.get('parameters', 'human_mode')))

    experiments = bool(ast.literal_eval(config.get('parameters', 'experiments')))
    actor_learning_rates = list(ast.literal_eval(config.get('parameters', 'actor_learning_rates')))
    critic_learning_rates = list(ast.literal_eval(config.get('parameters', 'critic_learning_rates')))
    iterations = int(config.get('parameters', 'iterations'))
    win_multipliers = list(ast.literal_eval(config.get('parameters', 'win_multipliers')))
    epsilon_functions = list(ast.literal_eval(config.get('parameters', 'epsilon_functions')))

    action_reward = float(config.get('parameters', 'action_reward'))
    win_reward = int(config.get('parameters', 'win_reward'))
    loss_reward = int(config.get('parameters', 'loss_reward'))
    win_multiplier = int(config.get('parameters', 'win_multiplier'))
    board_type = int(config.get('parameters', 'board_type'))
    board_size = int(config.get('parameters', 'board_size'))
    empty_nodes = list(ast.literal_eval(config.get('parameters', 'empty_nodes')))
    episodes = int(config.get('parameters', 'episodes'))
    test_episodes = int(config.get('parameters', 'test_episodes'))

    nn_critic = bool(config.get('parameters', 'nn_critic'))
    nn_dimentions = list(config.get('parameters', 'nn_dimentions'))

    actor_learning_rate = float(config.get('parameters', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('parameters', 'critic_learning_rate'))
    actor_decay_rate = float(config.get('parameters', 'actor_decay_rate'))
    critic_decay_rate = float(config.get('parameters', 'critic_decay_rate'))
    actor_discount_factor = float(config.get('parameters', 'actor_discount_factor'))
    critic_discount_factor = float(config.get('parameters', 'critic_discount_factor'))

    linear_epsilon = bool(ast.literal_eval(config.get('parameters', 'linear_epsilon')))
    epsilon = float(config.get('parameters', 'epsilon'))
    epsilon_decay = float(config.get('parameters', 'epsilon_decay'))

    visualize = str(config.get('parameters', 'visualize'))
    visualization_frame_delay = int(config.get('parameters', 'visualization_frame_delay'))

    def get_agent_params() -> tuple:
        return Config.actor_learning_rate, Config.critic_learning_rate, Config.actor_decay_rate, Config.critic_decay_rate, Config.actor_discount_factor, Config.critic_discount_factor
