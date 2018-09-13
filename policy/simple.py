"""
Simple (Policy Gradient) agents policy generator
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# TODO include TEAM1_UAV & TEAM2_UAV
# features in observation
INVISIBLE = -1
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
        self.round = 0
        self.update_freq = 1

        self.gamma = 0.99
        self.reward = []

        self.history = []
        for i in range(len(agent_list)):
            self.history.append([])

        tf.reset_default_graph()

        self.state_in = tf.placeholder(shape=[None,len(free_map),len(free_map[0]),5], dtype=tf.float32)
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
            self.free_map = free_map

        action_out = []
        for idx,agent in enumerate(agent_list):
            if agent.isAlive:
                x,y = agent.get_loc()
                curr_state = self.parse_obs(obs, x, y)

                action = self.sess.run(self.chosen_action, feed_dict={self.state_in:[curr_state]})
                action_out.append(action[0])

                self.history[idx].append([curr_state, action])
            else:
                action_out.append(STAY)

        return action_out

    def record_reward(self, reward):
        self.reward.append(reward)

    def update_network(self):
        self.round += 1

        states, actions, rewards = self.prepare_info()
        self.clear_record()

        feed_dict = {self.state_in:states, self.action_holder:actions, self.reward_holder:rewards}
        grads = self.sess.run(self.gradients, feed_dict=feed_dict)
        for idx,grad in enumerate(grads):
            self.gradBuffer[idx] += grad

        if self.round % self.update_freq == 0:
            feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
            _ = self.sess.run(self.update_batch, feed_dict=feed_dict)
            for ix,grad in enumerate(self.gradBuffer):
                self.gradBuffer[ix] = grad*0

    def clear_record(self):
        for i in range(len(self.history)):
            self.history[i] = []
        self.reward = []

    def prepare_info(self):
        states = []
        actions = []
        rewards = []

        for i in range(len(self.history)):
            individual_history = np.array(self.history[i])
            states.append(individual_history[:,0])
            actions.append([item for sublist in individual_history[:,1] for item in sublist])
            rewards.append(self.discount_rewards(len(individual_history)))

        states = np.array([item for sublist in states for item in sublist])
        actions = np.array([item for sublist in actions for item in sublist])
        rewards = np.array([item for sublist in rewards for item in sublist])

        return states, actions, rewards

    def discount_rewards(self, step_count):
        discounted_rewards = np.zeros(step_count)
        running_add = 0

        for t in reversed(range(step_count)):
            running_add = running_add * self.gamma + self.reward[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def parse_obs(self, obs, x, y):
        # the channel number for different features
        # Channel 0: Visible (not -1): 1
        # Channel 1: Team1_Background VS Team2_Background: 1 VS -1
        # Channel 2: Team1_UGV VS Team2_UGV: 1 VS -1
        # Channel 3: Team1_Flag VS Team2_Flag: 1 VS -1
        # Channel 4: Obstacle + Everything out of Boundary: 1
        # Ignore DEAD
        switcher = {
            TEAM1_BACKGROUND:(1,  1),
            TEAM2_BACKGROUND:(1, -1),
            TEAM1_UGV:(2,  1),
            TEAM2_UGV:(2, -1),
            TEAM1_FLAG:(3,  1),
            TEAM2_FLAG:(3, -1),
            OBSTACLE:(4,  1)
        }

        parsed_obs = np.zeros((len(obs),len(obs[0]),5))

        # Shift the active unit to the center of the observation
        x_shift = int(len(obs)/2 - x)
        y_shift = int(len(obs[0])/2 - y)

        for i in range(max(0, int(x-len(obs)/2)), min(len(obs), int(x+len(obs)/2))):
            for j in range(max(0, int(y-len(obs[0]))/2), min(len(obs[0]), int(y+len(obs[0])/2))):

                if obs[i][j] != INVISIBLE:
                    parsed_obs[i+x_shift][j+y_shift][0] = 1
                    result = switcher.get(obs[i][j], 'nothing')
                    if result != 'nothing':
                        parsed_obs[i+x_shift][j+y_shift][result[0]] = result[1]


        # add padding to Channel 4 for everything out of boundary
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                ori_i, ori_j = i - x_shift, j - y_shift
                if ori_i < 0 or ori_i >= len(obs) or ori_j < 0 or ori_j >= len(obs[i]):
                    parsed_obs[i][j][4] = 1

        return parsed_obs
