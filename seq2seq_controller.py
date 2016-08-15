'''
Created on 15 August 2016
@author: Kolesnikov Sergey
'''

import gym
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import rnn_cell as rnn_cell
from tensorflow.python.ops import seq2seq as seq2seq


result_location = './result'

# Set random seed
np.random.seed(42)

class Seq2seqController(object):
    def __init__(self, sess,
                 cell_type='rnn',
                 n_input=4, n_hidden=12, n_output=1, n_layers=2,
                 seq_len=1,
                 init_stddev=0.1):

        if cell_type == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif cell_type == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif cell_type == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(cell_type))

        self.sess = sess

        self.cell_type = cell_type
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.seq_len = seq_len

        cell = cell_fn(self.n_hidden)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * self.n_layers)

        self.input = tf.placeholder(tf.float32,
                                    [None, self.n_input])
        self.output = tf.placeholder(tf.float32,
                                     [None, self.n_output])
        self.initial_state = tf.Variable(
            cell.zero_state(1, tf.float32),
            trainable=False,
            name='init_state')

        initializer = tf.random_normal_initializer(stddev=init_stddev)

        with tf.variable_scope("seq2seq", reuse=False) as rnn_scope:
            W_hy = tf.get_variable("W_hy", [self.n_hidden, self.n_output],
                                   tf.float32, initializer=initializer)
            b_y = tf.get_variable("b_y", [self.n_output],
                                  tf.float32, initializer=initializer)

        with tf.variable_scope("seq2seq", reuse=True) as rnn_scope:
            W_hy = tf.get_variable("W_hy", [self.n_hidden, self.n_output],
                                   tf.float32)
            b_y = tf.get_variable("b_y", [self.n_output],
                                  tf.float32)

        last_state = self.initial_state

        outputs, last_state = seq2seq.rnn_decoder([self.input],
                                                  last_state,
                                                  self.cell, scope='seq2seq')
        self.final_state = last_state
        output = tf.reshape(tf.concat(1, outputs), [-1, self.n_hidden])
        self.logits = tf.matmul(output, W_hy) + b_y
        self.probs = tf.nn.softmax(self.logits)
        self.log_probs = tf.log(self.probs)

        self.action = tf.reshape(tf.multinomial(self.logits, 1), [])
        self.acts = tf.placeholder(tf.int32)
        self.rewards = tf.placeholder(tf.float32)
        # import pdb; pdb.set_trace()
        cycle_op = tf.assign(self.initial_state, last_state)
        self.cycle_op = tf.group(cycle_op, name='cyclope')

        # loss = seq2seq.sequence_loss_by_example(
        #     logits=[self.logits],
        #     targets=[tf.reshape(self.output, [-1])],
        #     weights=[tf.ones([1 * self.seq_len])])

        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(self.log_probs)[0]) * \
                        tf.shape(self.log_probs)[1] + self.acts
        act_prob = tf.gather(tf.reshape(self.log_probs, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.mul(act_prob, self.rewards))

        cost = tf.reduce_sum(loss)
        self.cost = cost / self.seq_len
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.update = optimizer.apply_gradients(zip(grads, tvars))

        def act(self, observation):
            # get one action, by sampling
            return self.sess.run(self.action,
                               feed_dict={self.input: [observation]})

        def train_step(self, obs, acts, rewards):
            train_feed = { self.input: obs,
                           self.acts: acts,
                           self.rewards: rewards }
            self.sess.run(self.update, feed_dict=batch_feed)

def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews

def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)

def main():

    env = gym.make('CartPole-v0')

    env.monitor.start(result_location, force=True)

    params = {
        'input_size': env.observation_space.shape[0],
        'hidden_size': 36,
        'num_actions': env.action_space.n,
        'learning_rate': 0.1
    }

    # environment params
    eparams = {
        'num_batches': 40,
        'ep_per_batch': 10
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        agent = Seq2seqController(sess,
                                  n_input=params['input_size'],
                                  n_hidden=params['hidden_size'],
                                  n_output=params['num_actions'])
        sess.run(tf.initialize_all_variables())

        for batch in range(eparams['num_batches']):

            print ('=====\nBATCH {}\n===='.format(batch))
            b_obs, b_acts, b_rews = [], [], []

            for _ in range(eparams['ep_per_batch']):

                obs, acts, rews = policy_rollout(env, agent)

                print ('Episode steps: {}'.format(len(obs)))

                b_obs.extend(obs)
                b_acts.extend(acts)

                rewards = process_rewards(rews)
                b_rews.extend(rewards)

            # update policy
            # normalize rewards; don't divide by 0
            b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

            agent.train_step(b_obs, b_acts, b_rews)

        env.monitor.close()


if __name__ == "__main__":
    main()
