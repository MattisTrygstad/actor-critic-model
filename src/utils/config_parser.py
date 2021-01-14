
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    board_type = str(config.get('parameters', 'board_type'))
    size = int(config.get('parameters', 'size'))
    open_cells = int(config.get('parameters', 'empty_nodes'))
    episodes = int(config.get('parameters', 'episodes'))
    nn_critic = bool(config.get('parameters', 'nn_critic'))
    nn_dimentions = list(config.get('parameters', 'nn_dimentions'))
    actor_learning_rate = float(config.get('parameters', 'actor_learning_rate'))
    critic_learning_rate = float(config.get('parameters', 'critic_learning_rate'))
    actor_decay_rate = float(config.get('parameters', 'actor_decay_rate'))
    critic_decay_rate = float(config.get('parameters', 'critic_decay_rate'))
    actor_discount_factor = float(config.get('parameters', 'actor_discount_factor'))
    critic_discount_factor = float(config.get('parameters', 'critic_discount_factor'))
    e_initial = float(config.get('parameters', 'e_initial'))
    e_decay = float(config.get('parameters', 'e_decay'))
    visualize = str(config.get('parameters', 'visualize'))
    visualization_frame_delay = int(config.get('parameters', 'visualization_frame_delay'))
