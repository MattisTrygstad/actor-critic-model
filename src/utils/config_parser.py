
import ast
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    human_mode = bool(ast.literal_eval(config.get('parameters', 'human_mode')))
    action_reward = float(config.get('parameters', 'action_reward'))
    win_reward = int(config.get('parameters', 'win_reward'))
    loss_reward = int(config.get('parameters', 'loss_reward'))
    board_type = int(config.get('parameters', 'board_type'))
    board_size = int(config.get('parameters', 'board_size'))
    empty_nodes = list(ast.literal_eval(config.get('parameters', 'empty_nodes')))
    episodes = int(config.get('parameters', 'episodes'))
    nn_critic = bool(config.get('parameters', 'nn_critic'))
    nn_dimentions = list(config.get('parameters', 'nn_dimentions'))
    actor_learning_rate = float(config.get('parameters', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('parameters', 'critic_learning_rate'))
    actor_decay_rate = float(config.get('parameters', 'actor_decay_rate'))
    critic_decay_rate = float(config.get('parameters', 'critic_decay_rate'))
    actor_discount_factor = float(config.get('parameters', 'actor_discount_factor'))
    critic_discount_factor = float(config.get('parameters', 'critic_discount_factor'))
    epsilon = float(config.get('parameters', 'epsilon'))
    #epsilon_decay = float(config.get('parameters', 'epsilon_decay'))
    visualize = str(config.get('parameters', 'visualize'))
    visualization_frame_delay = int(config.get('parameters', 'visualization_frame_delay'))
