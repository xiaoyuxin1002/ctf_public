import sys
import time
import gym
import gym_cap
import math
import numpy as np
from pathlib import Path

# the modules that you can use to generate the policy.
import policy.random
import policy.simple
import policy.roomba
import policy.simple_red

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

if Path('reward_records.txt').is_file():
    reward_file = open('reward_records.txt', 'r')
    rewards = reward_file.read().splitlines()
    total_score = np.sum([float(item) for item in rewards])
else:
    total_score = 0

# if Path('time_records.txt').is_file():
#     time_file = open('time_records.txt', 'r')
#     times = time_file.read().splitlines()
#     extra_time = np.sum([float(item) for item in times])
# else:
#     extra_time = 0

# reset the environment and select the policies for each of the team
policy_blue=policy.simple.PolicyGen(env.get_map, env.get_team_blue)
policy_red=policy.simple_red.PolicyGen(env.get_map, env.get_team_red)
observation = env.reset(map_size=20,
                        render_mode="env",
                        policy_blue=policy_blue,
                        policy_red=policy_red)

while True:

    prev_reward = 0
    done = False
    t = 0
    #curr_round_start_time = time.time()
    #print(env._env)

    while not done:

        #you are free to select a random action
        # or generate an action using the policy
        # or select an action manually
        # and the apply the selected action to blue team
        # or use the policy selected and provided in env.reset
        #action = env.action_space.sample()  # choose random action
        #action = policy_blue.gen_action(env.team1,observation,map_only=env.team_home)
        #action = [0, 0, 0, 0]
        #observation, reward, done, info = env.step(action)

        policy_blue.get_full_picture(env._env)
        policy_red.get_full_picture(env._env)
        observation, reward, done, info = env.step()  # feedback from environment

        policy_blue.record_reward(reward - prev_reward)#math.log(max(t-200, 1), 2))
        prev_reward = reward

        # render and sleep are not needed for score analysis
        # env.render(mode="fast")
        # time.sleep(.05)

        t += 1
        if t == 300:
            break

    #curr_round_time = time.time() - curr_round_start_time

    policy_blue.update_network(reward, t)#curr_round_time)


    round = policy_blue.sess.run(policy_blue.round)
    if round % 100 == 0:
        policy_blue.save_model()

    reward_file = open("reward_records.txt", "a")
    reward_file.write("%f\n" % reward)
    reward_file.close()

    time_file = open("step_records.txt", "a")
    time_file.write("%f\n" % t)
    time_file.close()

    total_score += reward
    env.reset()
    print("Round: %s, total time: %s s, score: %s" % (round, (time.time() - start_time), total_score))

    if round >= 40000:
         break
