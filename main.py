import argparse
import numpy as np
from PIL import Image
from environment import AtariGame
from nature_dqn_agent import NatureDQNAgent

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def preprocess(arr):
    img = Image.fromarray(arr)
    img = img.convert('L')
    img = img.resize((84, 84), Image.LANCZOS)
    return np.asarray(img, dtype=np.int32)


def main(args):
    atari = AtariGame('rom/Breakout.bin', args.random_seed, args.action_repeat)
    args.num_action = len(atari.get_action_space())
    args.image_width, args.image_height = 84, 84
    agent = NatureDQNAgent(args)

    state = preprocess(atari.get_image())
    agent.init_observe(state)

    while True:
        action = agent.get_action()
        reward = atari.act(action)
        state = preprocess(atari.get_image())
        terminated = atari.is_game_over()
        if terminated:
            atari.reset()

        agent.observe(state, action, reward, terminated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        'Human-level control through deep reinforcement learning')
    parser.add_argument('--random_seed', type=int, default=123, help=\
        'Random seed used to init Atari.')
    parser.add_argument('--minibatch_size', type=int, default=32, help=\
        'Number of trainning cases over which each stochastic gradient '\
        'descent (SGD) update is computed.')
    parser.add_argument('--reply_memory_size', type=int, default=1000000, help=\
        'SGD updates are sampled from this number of most recent frames.')
    parser.add_argument('--agent_history_length', type=int, default=4, help=\
        'The number of most recent frames experienced by the agent that are '\
        'given as input to the Q network.')
    parser.add_argument('--target_network_update_frequency', type=int, default=10000, help=\
        'The frequency (measured in the number of parameter updates) with '\
        'which the target network is updated.')
    parser.add_argument('--discount_factor', type=float, default=0.99, help=\
        'Discount factor gamma used in the Q-learning update.')
    parser.add_argument('--action_repeat', type=int, default=4, help=\
        'Repeat each action selected by the agent this many times. Using '\
        'a value of 4 results in the agent seeing only 4th input frame.')
    parser.add_argument('--update_frequency', type=int, default=4, help=\
        'The number of actions selected by the agent between successive '\
        'SGD updates. Using a value of 4 results in the agent selecting 4 '\
        'actions between each pair of successive update.')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help=\
        'The learning rate used by RMSProp.')
    parser.add_argument('--gradient_momentum', type=float, default=0.95, help=\
        'Gradient momentum used by RMSProp.')
    parser.add_argument('--min_squared_gradient', type=float, default=0.01, help=\
        'Constant added to the squared gradient in the denominator of the RMSProp update.')
    parser.add_argument('--initial_exploration', type=float, default=1.0, help=\
        'Initial value of epsilon in epsilon-greedy exploration.')
    parser.add_argument('--final_exploration', type=float, default=0.1, help=\
        'Final value in epsilon in epsilon-greedy exploration.')
    parser.add_argument('--final_exploration_frame', type=int, default=1000000, help=\
        'The number of frames over which the initial value of epsilon is linearly '\
        'annealed to its final value.')
    parser.add_argument('--replay_start_size', type=int, default=50000, help=\
        'A uniform random policy is run for this number of frames before learning '\
        'starts and the resulting experience is used to populate the replay memory.')
    parser.add_argument('--noop_max', type=int, default=30, help=\
        'Maximum number of "do nothing" actions to be performed by the agent at the '\
        'start of an episode.')
    args = AttrDict(vars(parser.parse_args()))
    main(args)
