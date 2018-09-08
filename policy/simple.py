"""
Simple (Policy Gradient) agents policy generator
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# TODO include TEAM1_UAV & TEAM2_UAV
# features in observation
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM2_UGV = 4
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
DEAD = 9

# actions
STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
ACTION_SPACE = [STAY, UP, RIGHT, DOWN, LEFT]

class PolicyGen:

    def __init__(self, free_map, agent_list):
        self.free_map = free_map
        self.round = 1
        self.update_freq = 2
        self.gamma = 0.99

        self.history = []
        for i in range(len(agent_list)):
            self.history.append([])

        tf.reset_default_graph()

        self.state_in = tf.placeholder(shape=[None,20,20,9], dtype=tf.int32)
        net = slim.conv2d(self.state_in, 64, [10,10])
        net = slim.max_pool2d(net, [2,2])
        net = slim.dropout(net, keep_prob=0.9)
        net = slim.conv2d(net, 32, [5,5], padding='VALID')
        net = slim.max_pool2d(net, [2,2])
        net = slim.flatten(net)
        self.output = slim.fully_connected(net,len(ACTION_SPACE),activation_fn=tf.nn.softmax)

        self.chosen_action = tf.argmax(self.output, 1)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0])*tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.gradBuffer = self.sess.run(tf.trainable_variables())
        for idx,grad in enumerate(self.gradBuffer):
            self.gradBuffer[idx] = grad*0


    def gen_action(self, agent_list, obs, free_map=None):

        if free_map is not None:
            self.update_network()

            self.free_map = free_map

            self.round = self.round + 1
            self.history = []
            for i in range(len(agent_list)):
                self.history.append([])

        curr_state = self.parse_obs(obs)

        action_out = []
        for idx,agent in enumerate(agent_list):

            x,y = agent.get_loc()
            my_loc = np.zeros((20,20,1))
            my_loc[x][y][0] = 1
            curr_state = np.append(my_loc, curr_state, axis=2)

            action = self.sess.run(self.chosen_action, feed_dict={self.state_in:curr_state})
            action_out.append(action)

            self.history[idx].append([curr_state, action, agent.individual_reward(obs)])

        return action_out

    def update_network(self):
        states, actions, rewards = self.extract_info()
        feed_dict = {self.state_in:states, self.action_holder:actions, self.reward_holder:rewards}
        grads = self.sess.run(self.gradients, feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        if self.round % self.update_freq == 0 and self.round != 0:
            feed_dict = dict(zip(self.gradient_holders, gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad*0

    def extract_info(self):
        states = acitons = rewards = []
        for i in range(len(self.history)): # len(agent_list), agent by agent
            for j in range(len(self.history[i])): # number of times gen_action() being called
                states.append(self.history[i][j][0])
                actions.append(self.history[i][j][1])
                rewards.append(self.history[i][j][2])
        return np.array(states), np.array(actions), self.discount_rewards(rewards)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discount_rewards[t] = running_add
        return discount_rewards

    def parse_obs(self, obs):
        # the channel number for each feature
        switcher = {
            TEAM1_BACKGROUND:0,
            TEAM2_BACKGROUND:1,
            TEAM1_UGV:2,
            TEAM2_UGV:3,
            TEAM1_FLAG:4,
            TEAM2_FLAG:5,
            OBSTACLE:6,
            DEAD:7
        }

        parsed_obs = np.zeros((20,20,8))

        for i in range(len(obs)):
            for j in range(len(obs[i])):
                if obs[i][j] != -1:
                    channel = switcher.get(obs[i][j], 'nothing')
                    if channel != 'nothing':
                        parsed_obs[i][j][channel] = 1

        return parsed_obs
