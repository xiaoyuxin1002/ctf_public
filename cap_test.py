import time
import gym
import gym_cap
import math
import numpy as np


# the modules that you can use to generate the policy.
import policy.random
import policy.simple
import policy.roomba

start_time = time.time()
env = gym.make("cap-v0") # initialize the environment

total_score = 0

# reset the environment and select the policies for each of the team
policy_blue=policy.simple.PolicyGen(env.get_map, env.get_team_blue)
policy_red=policy.random.PolicyGen(env.get_map, env.get_team_red)
observation = env.reset(map_size=20,
                        render_mode="env",
                        policy_blue=policy_blue,
                        policy_red=policy_red)

while True:

    prev_reward = 0
    done = False
    t = 0

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
        observation, reward, done, info = env.step()  # feedback from environment

        policy_blue.record_reward(reward - prev_reward)#math.log(max(t-200, 1), 2))
        prev_reward = reward

        # render and sleep are not needed for score analysis
        # env.render(mode="fast")
        # time.sleep(.05)

        t += 1
        if t == 1000:
            break

    total_score += reward
    env.reset()
    print("Total time: %s s, score: %s" % ((time.time() - start_time),total_score))

    update_start_time = time.time()
    policy_blue.update_network(reward)
    print("Update Time: %s s" % (time.time() - update_start_time))

    round = policy_blue.sess.run(policy_blue.round)
    if round % 10 == 0:
        reward_file = open("reward_records.txt", "a")
        for reward_record in policy_blue.sess.run(policy_blue.reward_history)[-10:]:
            reward_file.write("%f\n" % reward_record)
        reward_file.close()

        output_file = open("mean_reward_records.txt", 'a')
        output_file.write("rounds: %i-%i, mean reward: %f\n" % (round-9, round, np.mean(policy_blue.sess.run(policy_blue.reward_history)[-10:])))
        output_file.close()
        policy_blue.save_model()

    if round >= 30000:
         break
