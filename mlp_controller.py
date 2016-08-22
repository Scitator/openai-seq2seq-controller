'''
Created on 15 August 2016
@author: Kolesnikov Sergey
'''

import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

result_location = './result'

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

# Set random seed
np.random.seed(42)


class MlpController(object):
    def __init__(self,
                 n_obs_space=4, n_act_space=2,
                 learning_rate=0.0001):
        self.n_obs_space = n_obs_space
        self.n_act_space = n_act_space
        self.lr = learning_rate

        self.timestep = 0
        self.buffer = deque()
        self.epsilon = INITIAL_EPSILON

        self._init_variables()
        self._build_graph()
        self._build_optimization()

    def _init_variables(self):
        self.obs_input = tf.placeholder(tf.float32, [None, self.n_obs_space])
        self.act_input = tf.placeholder(tf.float32, [None, self.n_act_space])
        self.rew_input = tf.placeholder(tf.float32, [None])

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_obs_space, 12])),
            'h2': tf.Variable(tf.random_normal([12, 12])),
            'out': tf.Variable(tf.random_normal([12, self.n_act_space]))
        }
        self.bias = {
            'b1': tf.Variable(tf.random_normal([12])),
            'b2': tf.Variable(tf.random_normal([12])),
            'out': tf.Variable(tf.random_normal([self.n_act_space]))
        }

    def _build_graph(self):
        layer_1 = tf.add(
            tf.matmul(self.obs_input, self.weights['h1']),
            self.bias['b1'])
        layer_1 = tf.nn.tanh(layer_1)

        layer_2 = tf.add(
            tf.matmul(layer_1, self.weights['h2']),
            self.bias['b2'])
        layer_2 = tf.nn.tanh(layer_2)

        self.q_val = tf.add(
            tf.matmul(layer_2, self.weights['out']),
            self.bias['out'])

    def _build_optimization(self):
        Q_action = tf.reduce_sum(tf.mul(self.q_val, self.act_input),
                                 reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.rew_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def memento(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.n_act_space)
        one_hot_action[action] = 1

        self.buffer.append(
            (state, one_hot_action, reward, next_state, done))

        if len(self.buffer) > REPLAY_SIZE:
            self.buffer.popleft()

        if len(self.buffer) > BATCH_SIZE:
            self._train()

    def _train(self):
        self.timestep += 1

        minibatch = random.sample(self.buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.q_val.eval(
            feed_dict={self.obs_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(
                    reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.rew_input: y_batch,
            self.act_input: action_batch,
            self.obs_input: state_batch
        })

    def action(self, state):
        Q_value = self.q_val.eval(feed_dict={self.obs_input: [state]})[0]
        return np.argmax(Q_value)

    def e_greedy_action(self, state):
        Q_value = self.q_val.eval(feed_dict={self.obs_input: [state]})[0]

        if np.random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.random.randint(0, self.n_act_space - 1)
        else:
            return np.argmax(Q_value)


def main():
    env = gym.make(ENV_NAME)
    agent = MlpController(n_obs_space=env.observation_space.shape[0],
                          n_act_space=env.action_space.n)

    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())

    for episode in range(EPISODE):
        state = env.reset()
        for step in range(STEP):
            action = agent.e_greedy_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memento(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                  ave_reward)
            if ave_reward >= 200:
                break


if __name__ == '__main__':
    main()
