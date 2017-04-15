import os
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import deque, namedtuple
import nnutils

ReplayMemoryItem = namedtuple('ReplayMemoryItem', ['state', 'action', 'reward', 'next_state', 'terminated'])

class NatureDQNAgent(object):
    def __init__(self, args):
        self.args = args
        self.build_graph()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.step = 0
        self.episode = 0
        self.sum_reward = 0
        self.epsilon = args.initial_exploration
        self.replay_memory = deque(maxlen=self.args.reply_memory_size)

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

    def build_graph(self):
        H, W, C = self.args.image_height, self.args.image_width, self.args.agent_history_length
        self.ph_states = tf.placeholder(tf.float32, [None, H, W, C], name='states')
        self.ph_action = tf.placeholder(tf.int32, [None], name='action')
        self.ph_ys = tf.placeholder(tf.float32, [None], name='y')

        # network
        self.behavior_q, behavior_params, _, _ = self.build_q_network(self.ph_states, 'q_behavior')
        self.target_q, target_params, _, _ = self.build_q_network(self.ph_states, 'q_target')
        self.copy_step = []
        for name in behavior_params.iterkeys():
            bp, tp = behavior_params[name], target_params[name]
            self.copy_step.append(tp.assign(bp))

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
        # opt = tf.train.RMSPropOptimizer(learning_rate=self.args.learning_rate)
        # opt = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)

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

    def copy_network(self):
        self.sess.run(self.copy_step)
        print 'step', self.step, 'network copied'

    def train_q(self):
        minibatch = random.sample(self.replay_memory, self.args.minibatch_size)
        batch_states = [data.state for data in minibatch]
        batch_actions = [data.action for data in minibatch]
        batch_rewards = [data.reward for data in minibatch]
        batch_next_states = [data.next_state for data in minibatch]

        qvalue = self.sess.run(self.target_q, feed_dict={self.ph_states: batch_next_states})
        batch_ys = []
        for i in xrange(0, self.args.minibatch_size):
            y = batch_rewards[i]
            if not minibatch[i].terminated:
                y += self.args.discount_factor * np.max(qvalue[i])
            batch_ys.append(y)

        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_step], feed_dict={
            self.ph_states: batch_states,
            self.ph_action: batch_actions,
            self.ph_ys: batch_ys
        })
        if self.step % 128 == 0:
            self.summary_writer.add_summary(summary, self.step)
            summary = tf.Summary(value=[tf.Summary.Value(tag='epsilon', simple_value=self.epsilon)])
            self.summary_writer.add_summary(summary, self.step)

        if self.step % 10000 == 0:
            self.saver.save(self.sess, 'models/nature-dqn', global_step=self.step)

    def observe(self, state, action, reward, terminated):
        self.step += 1
        next_state = np.concatenate((self.current_state[:, :, 1:], state[:, :, np.newaxis]), axis=2)
        assert (next_state[:, :, -1] == state).all()
        item = ReplayMemoryItem(state=self.current_state,
                                action=action,
                                reward=reward,
                                next_state=next_state,
                                terminated=terminated)
        self.replay_memory.append(item)
        if self.step > self.args.replay_start_size:
            if self.step % self.args.update_frequency == 0:
                self.train_q()
            if self.step % self.args.target_network_update_frequency == 0:
                self.copy_network()
        self.current_state = next_state
        self.sum_reward += reward

        if terminated:
            self.episode += 1
            summary = tf.Summary(value=[tf.Summary.Value(tag='episode reward', simple_value=self.sum_reward)])
            self.summary_writer.add_summary(summary, self.step)
            self.sum_reward = 0

        if self.args.replay_start_size < self.step <= self.args.final_exploration_frame:
            steps = self.args.final_exploration_frame - self.args.replay_start_size
            self.epsilon -= (self.args.initial_exploration - self.args.final_exploration) / steps

    def get_action(self):
        if random.random() < self.epsilon:
            action = random.randrange(self.args.num_action)
        else:
            feed = {self.ph_states: [self.current_state]}
            qvalue = self.sess.run(self.behavior_q, feed_dict=feed)[0]
            action = np.argmax(qvalue)
        return action

    def init_observe(self, state):
        self.current_state = np.stack([state] * self.args.agent_history_length, axis=2)


