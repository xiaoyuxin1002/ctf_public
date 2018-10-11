"""
Simple (Policy Gradient) agents policy generator
"""

import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pathlib import Path

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

        for i in range(len(free_map)):
            for j in range(len(free_map[0])):
                if free_map[i][j] == TEAM1_FLAG:
                    self.flag_loc = (i, j)
                    break

        # self.update_freq = 2
        #
        # self.gamma = 0.999
        # self.reward = []
        #
        # self.history = []
        # for i in range(len(agent_list)):
        #    self.history.append([])

        tf.reset_default_graph()

        # self.round = tf.Variable(0, trainable=False, name='round')
        # self.round_increment = tf.assign(self.round, self.round+1)

        # if Path('reward_records.txt').is_file():
        #     reward_file = open('reward_records.txt', 'r')
        #     rewards = reward_file.read().splitlines()
        #     self.reward_history = tf.constant([float(item) for item in rewards])
        # else:
        #     self.reward_history = tf.constant([])
        #
        # if Path('step_records.txt').is_file():
        #     step_file = open('step_records.txt', 'r')
        #     steps = reward_file.read().splitlines()
        #     self.step_history = tf.constant([float(item) for item in steps])
        # else:
        #     self.step_history = tf.constant([])
        #
        # self.curr_reward = tf.placeholder(tf.float32, shape=(), name="reward")
        # self.mean_reward_10 = tf.placeholder(tf.float32, shape=(), name="mean_reward_10")
        # self.mean_reward_100 = tf.placeholder(tf.float32, shape=(), name="mean_reward_100")
        #
        # self.curr_steps_taken = tf.placeholder(tf.float32, shape=(), name="steps_taken")
        # self.mean_steps_10 = tf.placeholder(tf.float32, shape=(), name="mean_steps_10")
        # self.mean_steps_100 = tf.placeholder(tf.float32, shape=(), name="mean_steps_100")
        #
        # tf.summary.scalar('reward', self.curr_reward)
        # tf.summary.scalar('mean_reward_10', self.mean_reward_10)
        # tf.summary.scalar('mean_reward_100', self.mean_reward_100)
        # tf.summary.scalar('step', self.curr_steps_taken)
        # tf.summary.scalar('mean_step_10', self.mean_steps_10)
        # tf.summary.scalar('mean_step_100', self.mean_steps_100)
        # self.merged_summary_op = tf.summary.merge_all()

        self.state_in = tf.placeholder(shape=[None,len(free_map),len(free_map[0]),5], dtype=tf.float32)
        net = slim.conv2d(self.state_in, 32, [5,5], padding='VALID')
        #net = slim.max_pool2d(net, [2,2])
        net = slim.dropout(net, keep_prob=0.9)
        net = slim.conv2d(net, 16, [3,3], padding='VALID')
        #net = slim.max_pool2d(net, [2,2])
        net = slim.flatten(net)
        net = slim.fully_connected(net, 256, activation_fn=tf.nn.softmax)
        #net = slim.fully_connected(net, 32, activation_fn=tf.nn.softmax)
        self.output = slim.fully_connected(net, len(ACTION_SPACE), activation_fn=tf.nn.softmax)

        #self.chosen_action = tf.argmax(self.output, 1)
        #self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        #self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # self.indexes = tf.range(0, tf.shape(self.output)[0])*tf.shape(self.output)[1] + self.action_holder
        # self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        #
        # self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        # tvars = tf.trainable_variables()
        # self.gradient_holders = []
        # for idx, var in enumerate(tvars):
        #     placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
        #     self.gradient_holders.append(placeholder)
        #
        # self.gradients = tf.gradients(self.loss, tvars)
        #
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        # self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        ckpt = tf.train.get_checkpoint_state('./model')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        # else:
        #     self.sess.run(tf.global_variables_initializer())

        # self.gradBuffer = self.sess.run(tf.trainable_variables())
        # for idx,grad in enumerate(self.gradBuffer):
        #     self.gradBuffer[idx] = grad*0

    # def save_model(self):
    #     self.saver.save(self.sess, './model/ctf_policy.ckpt', global_step=self.sess.run(self.round))

    def get_full_picture(self, _env):
        self._env = _env

    def gen_action(self, agent_list, obs, free_map=None):

        obs = self._env

        if free_map is not None:
            self.free_map = free_map
            for i in range(len(free_map)):
                for j in range(len(free_map[0])):
                    if free_map[i][j] == TEAM1_FLAG:
                        self.flag_loc = (i, j)
                        break

        action_out = []
        for idx,agent in enumerate(agent_list):
            if agent.isAlive:
                x,y = agent.get_loc()
                curr_state = self.parse_obs(obs, x, y)

                output = self.sess.run(self.output, feed_dict={self.state_in:[curr_state]})
                action = np.random.choice(ACTION_SPACE, p=output[0])

                action_out.append(self.flip_action(action))

                #self.history[idx].append([curr_state, action])
            else:
                action_out.append(STAY)

        return action_out

    def flip_action(self, action):
        if action == UP:
            return DOWN
        elif action == DOWN:
            return UP
        return action

    def parse_obs(self, obs, x, y):
        # the channel number for different features
        # Channel 0: INVISIBLE: 1
        # Channel 1: Team1_Background VS Team2_Background: -1 VS 1
        # Channel 2: Team1_UGV VS Team2_UGV: -1 VS 1
        # Channel 3: Team1_Flag VS Team2_Flag: -1 VS 1
        # Channel 4: Obstacle + Everything out of Boundary: 1
        # Ignore DEAD
        switcher = {
            INVISIBLE:(0,  1),
            TEAM1_BACKGROUND:(1, -1),
            TEAM2_BACKGROUND:(1,  1),
            TEAM1_UGV:(2, -1),
            TEAM2_UGV:(2,  1),
            TEAM1_FLAG:(3, -1),
            TEAM2_FLAG:(3,  1),
            OBSTACLE:(4,  1)
        }

        parsed_obs = np.zeros((len(obs),len(obs[0]),5))

        # Shift the active unit to the center of the observation
        x_shift = int(len(obs)/2 - x)
        y_shift = int(len(obs[0])/2 - y)

        for i in range(max(0, int(x-len(obs)/2)), min(len(obs), int(x+len(obs)/2))):
            for j in range(max(0, int(y-len(obs[0]))/2), min(len(obs[0]), int(y+len(obs[0])/2))):

                # if obs[i][j] != INVISIBLE:
                #     parsed_obs[i+x_shift][j+y_shift][0] = 1
                result = switcher.get(obs[i][j], 'nothing')
                if result != 'nothing':
                    parsed_obs[i+x_shift][j+y_shift][result[0]] = result[1]

        # add the background of the current location to channel 1
        if self.free_map[x][y] == TEAM1_BACKGROUND:
            parsed_obs[x+x_shift][y+y_shift][1] = -1
        else:
            parsed_obs[x+x_shift][y+y_shift][1] = 1

        # add the enemy flag location to channel 3
        flag_loc_x, flag_loc_y = self.flag_loc[0]+x_shift, self.flag_loc[1]+y_shift
        if flag_loc_x >= 0 and flag_loc_x < len(obs) and flag_loc_y >= 0 and flag_loc_y < len(obs[0]):
            #parsed_obs[flag_loc_x][flag_loc_y][0] = 1
            parsed_obs[flag_loc_x][flag_loc_y][1] = -1
            parsed_obs[flag_loc_x][flag_loc_y][3] = -1

        # add padding to Channel 4 for everything out of boundary
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                ori_i, ori_j = i - x_shift, j - y_shift
                if ori_i < 0 or ori_i >= len(obs) or ori_j < 0 or ori_j >= len(obs[i]):
                    parsed_obs[i][j][4] = 1

        return parsed_obs

    # def record_reward(self, reward):
    #     self.reward.append(reward)

    # def update_network(self, reward, steps_taken):
    #     states, actions, rewards = self.prepare_info()
    #     self.clear_record()
    #
    #     feed_dict = {self.state_in:states, self.action_holder:actions, self.reward_holder:rewards}
    #     grads = self.sess.run(self.gradients, feed_dict=feed_dict)
    #     for idx,grad in enumerate(grads):
    #         self.gradBuffer[idx] += grad
    #
    #     self.sess.run(self.round_increment)
    #     round = self.sess.run(self.round)
    #     if round % self.update_freq == 0:
    #         feed_dict = dict(zip(self.gradient_holders, self.gradBuffer))
    #         _ = self.sess.run(self.update_batch, feed_dict=feed_dict)
    #         for ix,grad in enumerate(self.gradBuffer):
    #             self.gradBuffer[ix] = grad*0
    #
    #     self.reward_history = tf.concat([self.reward_history, [float(reward)]], 0)
    #     self.step_history = tf.concat([self.step_history, [float(steps_taken)]], 0)
    #
    #     feed_dict = { \
    #         self.curr_reward:reward, \
    #         self.mean_reward_10:np.mean(self.sess.run(self.reward_history)[-10:]), \
    #         self.mean_reward_100:np.mean(self.sess.run(self.reward_history)[-100:]), \
    #         self.curr_steps_taken:steps_taken, \
    #         self.mean_steps_10:np.mean(self.sess.run(self.step_history)[-10:]), \
    #         self.mean_steps_100:np.mean(self.sess.run(self.step_history)[-100:]) \
    #     }
    #     summary = self.sess.run(self.merged_summary_op, feed_dict=feed_dict)
    #     self.writer.add_summary(summary, global_step=self.sess.run(self.round))

    # def clear_record(self):
    #     for i in range(len(self.history)):
    #         self.history[i] = []
    #     self.reward = []

    # def prepare_info(self):
    #     states = []
    #     actions = []
    #     rewards = []
    #
    #     for i in range(len(self.history)):
    #         individual_history = np.array(self.history[i])
    #         states.append(individual_history[:,0])
    #         actions.append(individual_history[:,1])
    #         rewards.append(self.discount_rewards(len(individual_history)))
    #
    #     states = np.array([item for sublist in states for item in sublist])
    #     actions = np.array([item for sublist in actions for item in sublist])
    #     rewards = np.array([item for sublist in rewards for item in sublist])
    #
    #     return states, actions, rewards

    # def discount_rewards(self, step_count):
    #     discounted_rewards = np.zeros(step_count)
    #     running_add = 0
    #
    #     for t in reversed(range(step_count)):
    #         running_add = running_add * self.gamma + self.reward[t]
    #         discounted_rewards[t] = running_add
    #         return discounted_rewards
