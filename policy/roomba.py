"""

Roomba agents policy generator.

"""
import numpy as np

class PolicyGen:

    def __init__(self, agent_list, map):
        self.free_map = map
        self.prev_action = np.random.choice([2,4], len(agent_list)).tolist()
        self.count = 100

    def gen_action(self, agent_list, obs):
        action_out = []

        for idx,agent in enumerate(agent_list):
            a = self.roomba(agent, idx, obs)
            action_out.append(a)

        return action_out

    def roomba(self, agent, index, obs):

        x,y = agent.get_loc()
        action = self.prev_action[index]

        self.count = self.count - 1

        # if cross the boundary, go back to avoid being killed
        if (y >= len(obs[0])):
            action = 1
            self.prev_action[index] = action
            return action

        # increase randomness in actions for exploration
        # and avoid the same pattern
        if (self.count <= 0):
            self.count = 100
            action = np.random.randint(1, 5)
            self.prev_action[index] = action
            return action

        if (action == 1 or action == 3):
            # if entering a new horizontal line, exploring this line
            possible_actions = []
            for possible_action in [2, 4]:
                next_x, next_y = self.next_position(x, y, possible_action)
                if (not self.check_obstacle(next_x, next_y, obs)):
                    possible_actions.append(possible_action)
            if (possible_actions != []):
                for i in range(3):
                    action = np.random.choice(possible_actions)
                    if (not self.within_attack_range(agent.a_range, x, y, obs)):
                        self.prev_action[index] = action
                        return action
                return 0
            else:
                # if being traped in a corner, turn back
                if (action == 1):
                     action = 3
                     self.prev_action[index] = action
                     return action
                else:
                     action = 1
                     self.prev_action[index] = action
                     return action
        else:
            next_x, next_y = self.next_position(x, y, action)
            # continue exploring the current horizontal line if possible
            if (not self.check_obstacle(next_x, next_y, obs)):
                if (not self.within_attack_range(agent.a_range, x, y, obs)):
                    return action
                return 0
            else:
                # otherwist, change to a new line
                possible_actions = []
                for possible_action in [1, 3]:
                    next_x, next_y = self.next_position(x, y, possible_action)
                    if (not self.check_obstacle(next_x, next_y, obs) and
                       not self.within_attack_range(agent.a_range, x, y, obs)):
                        possible_actions.append(possible_action)
                if (possible_actions != []):
                    for i in range(3):
                        action = np.random.choice(possible_actions)
                        if (not self.within_attack_range(agent.a_range, x, y, obs)):
                            self.prev_action[index] = action
                            return action
                    return 0
                else:
                    # if being traped in a corner, turn back
                    if (action == 2):
                         action = 4
                         self.prev_action[index] = action
                         return action
                    else:
                         action = 2
                         self.prev_action[index] = action
                         return action

    def next_position(self, x, y, action):
        # 0:stay
        if (action == 0):
            return x, y
        # 1:up
        elif (action == 1):
            return x, y-1
        # 2:right
        elif (action == 2):
            return x+1, y
        # 3:down
        elif (action == 3):
            return x, y+1
        # 4:left
        else:
            return x-1, y

    def check_obstacle(self, next_x, next_y, obs):
        # out of bound
        if (next_x < 0 or next_x >= len(obs) or
           next_y < 0 or next_y >= len(obs[0])):
            return True

        # obstacle
        if (obs[next_x][next_y] == 8):
            return True

        # ally
        if (obs[next_x][next_y] == 2):
            return True

        return False

    def within_attack_range(self, a_range, x, y, obs):
        # within my own territory
        if (obs[x][y] == 0):
            return False

        for delta_x in range(-2*a_range, 2*a_range+1):
            for delta_y in range(-2*a_range, 2*a_range+1):
                loc_x, loc_y = x+delta_x, y+delta_y
                if (delta_x**2 + delta_y**2 <= (a_range*2)**2 and
                    (not (loc_x<0 or loc_y<0 or loc_x>=len(obs) or loc_y>=len(obs[0]))) and
                    obs[loc_x][loc_y]==4):
                    return True
        return False
