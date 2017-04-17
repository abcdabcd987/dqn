import os
import time
import random
import argparse
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime
import nnutils
from utils import AttrDict
from environment import AtariGame

class AsyncDQN(object):
    def __init__(self, args):
        self.args = args
        self.global_steps = [0] * self.args.num_workers
        self.build_graph()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        logdir = os.path.join('logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(logdir)
        self.summary_writer = tf.summary.FileWriter(logdir, self.sess.graph, flush_secs=2)
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("models")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "model loaded:", checkpoint.model_checkpoint_path
        else:
            print "no model found"
    
    def inc_global_step(self, worker_id):
        self.global_steps[worker_id] += 1
    
    @property
    def global_step(self):
        return np.sum(self.global_steps, dtype=int)

    def build_graph(self):
        H, W, C = self.args.image_height, self.args.image_width, self.args.agent_history_length
        self.ph_states = tf.placeholder(tf.float32, [None, H, W, C], name='states')
        self.ph_action = tf.placeholder(tf.int32, [None], name='action')
        self.ph_ys = tf.placeholder(tf.float32, [None], name='y')

        # network
        self.behavior_q, behavior_params, _, _ = self.build_q_network(self.ph_states, 'q_behavior')
        self.target_q, target_params, _, _ = self.build_q_network(self.ph_states, 'q_target')
        self.copy_op = []
        for name in behavior_params.iterkeys():
            bp, tp = behavior_params[name], target_params[name]
            self.copy_op.append(tp.assign(bp))

        # loss
        action_one_hot = tf.one_hot(self.ph_action, self.args.num_action)
        qvalue = tf.reduce_sum(self.behavior_q * action_one_hot, reduction_indices=1)
        delta = self.ph_ys - qvalue
        clipped_delta = tf.clip_by_value(delta, clip_value_min=-1.0, clip_value_max=1.0)
        self.loss = tf.reduce_mean(tf.square(clipped_delta))

        # optimizer
        opt = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate,
                                        momentum=self.args.gradient_momentum,
                                        epsilon=self.args.min_squared_gradient)

        self.train_op = opt.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        self.summary_step = tf.summary.merge_all()

    def build_q_network(self, x, scope):
        with tf.variable_scope(scope):
            layers = [
                nnutils.InputLayer(),
                nnutils.Conv2DLayer('conv1', ksize=(8, 8), kernels=32, strides=(4, 4), padding='VALID', act=tf.nn.relu),
                nnutils.Conv2DLayer('conv2', ksize=(4, 4), kernels=64, strides=(2, 2), padding='VALID', act=tf.nn.relu),
                nnutils.Conv2DLayer('conv3', ksize=(3, 3), kernels=64, strides=(1, 1), padding='VALID', act=tf.nn.relu),
                nnutils.FlattenLayer(),
                nnutils.FCLayer('fc1', dim=512, act=tf.nn.relu),
                nnutils.FCLayer('fc2', dim=self.args.num_action, act=None)
            ]
            out, l1_loss, l2_loss = nnutils.forward_all_layers(layers, x)
            params = nnutils.get_all_parameters(layers)
            pdict = { t.name.replace(scope, ''): t for t in params }
            return out, pdict, l1_loss, l2_loss

    def learner_thread(self, worker_id, stop_flag):
        env = AtariGame('rom/Breakout.bin', args.random_seed + worker_id, args.action_repeat)
        tc = AttrDict()
        tc.worker_id = worker_id
        tc.batch_states = []
        tc.batch_actions = []
        tc.batch_rewards = []
        tc.batch_terminated = []
        tc.batch_next_states = []
        tc.initial_epsilon = 1.0
        tc.final_epsilon = np.random.choice([.1,.01,.5], 1, p=[0.4,0.3,0.3])[0]
        tc.epsilon = tc.initial_epsilon
        tc.episode = 0
        tc.step = 0
        tc.sum_reward = 0
        time.sleep(0.2 * worker_id)

        while not stop_flag.is_set():
            env.reset(noop=random.randrange(self.args.noop_max))
            state = env.get_processed_image()
            tc.current_state = np.stack([state] * self.args.agent_history_length, axis=2)
            terminated = env.is_game_over()
            while not terminated:
                action = self.get_action(tc.current_state, tc.epsilon)
                reward = np.clip(env.act(action), -1.0, 1.0)
                state = env.get_processed_image()
                terminated = env.is_game_over()
                self.observe(tc, state, action, reward, terminated)

    def train_q(self, tc):
        qvalue = self.sess.run(self.target_q, feed_dict={self.ph_states: tc.batch_next_states})
        batch_ys = []
        for i in xrange(0, len(tc.batch_rewards)):
            y = tc.batch_rewards[i]
            if not tc.batch_terminated[i]:
                y += self.args.discount_factor * np.max(qvalue[i])
            batch_ys.append(y)

        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_step], feed_dict={
            self.ph_states: tc.batch_states,
            self.ph_action: tc.batch_actions,
            self.ph_ys: batch_ys
        })
        global_step = self.global_step
        if global_step % 128 == 0:
            self.summary_writer.add_summary(summary, global_step)
            summary = tf.Summary(value=[tf.Summary.Value(tag='epsilon', simple_value=tc.epsilon)])
            self.summary_writer.add_summary(summary, global_step)

    def observe(self, tc, state, action, reward, terminated):
        self.inc_global_step(tc.worker_id)
        tc.step += 1
        next_state = np.concatenate((tc.current_state[:, :, 1:], state[:, :, np.newaxis]), axis=2)
        # assert (next_state[:, :, -1] == state).all()

        tc.batch_states.append(tc.current_state)
        tc.batch_actions.append(action)
        tc.batch_rewards.append(reward)
        tc.batch_terminated.append(terminated)
        tc.batch_next_states.append(next_state)

        if self.global_step % self.args.target_network_update_frequency == 0:
            self.sess.run(self.copy_op)
            self.saver.save(self.sess, 'models/nature-dqn', global_step=self.global_step)
        if tc.step % self.args.update_frequency == 0 or terminated:
            self.train_q(tc)
            del tc.batch_states[:]
            del tc.batch_actions[:]
            del tc.batch_rewards[:]
            del tc.batch_terminated[:]
            del tc.batch_next_states[:]
        tc.current_state = next_state
        tc.sum_reward += reward

        if terminated:
            tc.episode += 1
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode reward', simple_value=tc.sum_reward)])
            self.summary_writer.add_summary(summary, self.global_step)
            tc.sum_reward = 0

        if tc.step < self.args.final_exploration_frame:
            tc.epsilon -= (tc.initial_epsilon - tc.final_epsilon) / self.args.final_exploration_frame

    def get_action(self, current_state, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.args.num_action)
        else:
            feed = {self.ph_states: [current_state]}
            qvalue = self.sess.run(self.behavior_q, feed_dict=feed)[0]
            action = np.argmax(qvalue)
        return action
    
    def train(self):
        self.global_steps = [0] * self.args.num_workers
        stop_flag = threading.Event()
        workers = []
        for i in xrange(self.args.num_workers):
            t = threading.Thread(target=self.learner_thread, args=(i, stop_flag))
            t.start()
            workers.append(t)
        try:
            while True:
                time.sleep(60)
        except:
            print 'stopping...'
            stop_flag.set()
            for t in workers:
                t.join()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        'Asynchronous Methods for Deep Reinforcement Learning arXiv:1602.01783v2')
    parser.add_argument('--random_seed', type=int, default=123, help=\
        'Random seed used to init Atari.')
    parser.add_argument('--agent_history_length', type=int, default=4, help=\
        'The number of most recent frames experienced by the agent that are '\
        'given as input to the Q network.')
    parser.add_argument('--target_network_update_frequency', type=int, default=40000, help=\
        'The frequency (measured in the number of parameter updates) with '\
        'which the target network is updated.')
    parser.add_argument('--discount_factor', type=float, default=0.99, help=\
        'Discount factor gamma used in the Q-learning update.')
    parser.add_argument('--action_repeat', type=int, default=4, help=\
        'Repeat each action selected by the agent this many times. Using '\
        'a value of 4 results in the agent seeing only 4th input frame.')
    parser.add_argument('--update_frequency', type=int, default=5, help=\
        'The number of actions selected by the agent between successive '\
        'SGD updates. Using a value of 4 results in the agent selecting 4 '\
        'actions between each pair of successive update.')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help=\
        'The learning rate used by RMSProp.')
    parser.add_argument('--gradient_momentum', type=float, default=0.95, help=\
        'Gradient momentum used by RMSProp.')
    parser.add_argument('--min_squared_gradient', type=float, default=0.01, help=\
        'Constant added to the squared gradient in the denominator of the RMSProp update.')
    parser.add_argument('--final_exploration_frame', type=int, default=1000000, help=\
        'The number of frames over which the initial value of epsilon is linearly '\
        'annealed to its final value.')
    parser.add_argument('--noop_max', type=int, default=30, help=\
        'Maximum number of "do nothing" actions to be performed by the agent at the '\
        'start of an episode.')
    parser.add_argument('--num_workers', type=int, required=True)
    args = AttrDict(vars(parser.parse_args()))
    
    atari = AtariGame('rom/Breakout.bin', args.random_seed, args.action_repeat)
    args.num_action = len(atari.get_action_space())
    args.image_width, args.image_height = 84, 84
    del atari

    agent = AsyncDQN(args)
    agent.train()
