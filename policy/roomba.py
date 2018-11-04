"""Simple agents policy generator.

This module demonstrates an example of a simple heuristic policy generator
for Capture the Flag environment.
    http://github.com/osipychev/ctf_public/

DOs/Denis Osipychev
    http://www.denisos.com
"""
import numpy as np

Exploring_Group = 0
Patrolling_Group = 1
Patrolling_Rate = 0.5

# get from const.py
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM1_UAV = 3
TEAM2_UGV = 4
TEAM2_UAV = 5
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
DEAD = 9

# directions
STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
UNDEFINED = 5

class PolicyGen:

    """
    Roomba Policy:
    1. Split the team into two groups by an exploration ratio
    2. Group 1 is responsible for killing enemies entering my territory
        2.1 Goup 1 will approach the boundary and start to patrol along the boundary
        2.2 If the agent finds enemies within the attack range, go ahead and kill them
    3. Group 2 is responsible for capturing the enemy's flag
        3.1 Group 2 will go down first and enter the enemy's territory, then loop through:
            3.1.1 Check if the enemy's flag is nearby
                    if yes, go there, win the game and end the loop
                    if no, continue
            3.1.2 Check if enemies are nearby
                    if yes, avoid that direction
            3.1.3 Randomly select a direction among those left, favoring horizontal search to vertical search
    """

    def __init__(self, map, agent_list):
        self.map = map
        self.identity_list = np.zeros(len(agent_list))
        self.identity_list[0:int(len(agent_list)*Patrolling_Rate)] = Patrolling_Group
        self.prev_action_list = np.full((len(agent_list)), DOWN)
        # add randomness into action to prevent stucking in a local region
        self.count = 50

    def gen_action(self, agent_list, obs, free_map=None):
        if free_map is not None:
            self.map = free_map

        self.count = self.count - 1
        action_out = []

        for idx,agent in enumerate(agent_list):
            a = self.roomba(agent, idx, self.identity_list[idx], obs)
            action_out.append(a)

        if self.count == 0:
            self.count = 50

        return action_out

    def roomba(self, agent, index, identity, obs):
        if identity == Patrolling_Group:
            return self.patrol(agent, index, obs)
        else:
            return self.explore(agent, index, obs)

    def find_way_down(self, x, y, index, obs):
        # if it is possible to go down, then go down
        action = DOWN
        next_x, next_y = self.next_position(x, y, action)
        if not self.check_obstacle(next_x, next_y, obs):
            self.prev_action_list[index] = action
            return action
        else:
            # if it is possible continue the previous direction, then continue
            action = self.prev_action_list[index]
            if action == STAY: action = LEFT
            next_x, next_y = self.next_position(x, y, action)
            if not self.check_obstacle(next_x, next_y, obs):
                self.prev_action_list[index] = action
                return action
            # otherwise change direction
            else:
                potential_actions = [LEFT, UP]
                if action == LEFT:
                    potential_actions = [RIGHT, UP]
                elif action == DOWN:
                    potential_actions = [LEFT, RIGHT, UP]
                elif action == UP:
                    potential_actions = [LEFT, RIGHT]
                for action in potential_actions:
                    next_x, next_y = self.next_position(x, y, action)
                    if not self.check_obstacle(next_x, next_y, obs):
                        self.prev_action_list[index] = action
                        return action
                # TODO: need to solve the extreme case of being trapped in a concave shape
        return STAY

    def patrol(self, agent, index, obs):
        x, y = agent.get_loc()

        # if not along the boundary,
        # go down to approach the boundary
        if y < len(obs[0]) / 2 - 1:
            return self.find_way_down(x, y, index, obs)

        # if found enemies within attack range, go head and kill them
        action = self.find_attacking_direction(agent.a_range, x, y, obs)
        if action != UNDEFINED:
            self.prev_action_list[index] = action
            return action

        # otherwise, patrol along the boundary
        action = self.prev_action_list[index]
        if action == DOWN or action == STAY:
            action = np.random.choice([LEFT, RIGHT])

        next_x, next_y = self.next_position(x, y, action)
        if not self.check_obstacle(next_x, next_y, obs):
            self.prev_action_list[index] = action
            return action
        else:
            if action == LEFT: action = RIGHT
            elif action == RIGHT: action = LEFT
            next_x, next_y = self.next_position(x, y, action)
            if not self.check_obstacle(next_x, next_y, obs):
                self.prev_action_list[index] = action
                return action

        action = STAY
        self.prev_action_list[index] = action
        return action

    def explore(self, agent, index, obs):
        x, y = agent.get_loc()

        if y < len(obs[0]) / 2:
            return self.find_way_down(x, y, index, obs)

        # if enemy flag is nearby, go ahead and capture it
        action = self.find_enemy_flag_nearby(x, y, obs)
        if action != UNDEFINED:
            return action

        if self.count != 0:
            # check all the potential actions
            potential_actions = []
            for action in [UP, RIGHT, DOWN, LEFT]:
                next_x, next_y = self.next_position(x, y, action)
                if not self.check_obstacle_and_enemy(next_x, next_y, obs) and \
                    not self.check_dangerous(agent.a_range, next_x, next_y, obs):
                    potential_actions.append(action)

            # choose action among potential actions
            check_sequence = np.random.permutation([LEFT, RIGHT, DOWN, UP]).tolist()
            for action in check_sequence:
                if action in potential_actions:
                    self.prev_action_list[index] = action
                    return action

        # choose a random action, favoring horizontal search to vertical search
        action = np.random.choice([UP, RIGHT, DOWN, LEFT], p=[0.1, 0.35, 0.2, 0.35])
        self.prev_action_list[index] = action
        return action

    def next_position(self, x, y, action):
        if action == STAY:
            return x, y
        elif action == UP:
            return x, y-1
        elif action == RIGHT:
            return x+1, y
        elif action == DOWN:
            return x, y+1
        else:
            return x-1, y

    def check_out_of_bound(self, x, y, x_len, y_len):
        if x<0 or x>=x_len or y<0 or y>=y_len:
            return True
        return False

    def check_obstacle(self, x, y, obs):
        if self.check_out_of_bound(x, y, len(obs), len(obs[0])):
            return True
        if obs[x][y] == OBSTACLE:
            return True
        if obs[x][y] == TEAM1_UGV:
            return True
        return False

    def check_obstacle_and_enemy(self, x, y, obs):
        if self.check_obstacle(x, y, obs):
            return True
        if obs[x][y] == TEAM2_UGV:
            return True
        return False

    def find_attacking_direction(self, a_range, x, y, obs):
        for delta_x in range(-2*a_range, 2*a_range+1):
            for delta_y in range(-2*a_range, 2*a_range+1):
                loc_x, loc_y = x+delta_x, y+delta_y
                if delta_x**2 + delta_y**2 <= (a_range*2)**2 and \
                    not self.check_out_of_bound(loc_x, loc_y, len(obs), len(obs[0])) and \
                    obs[loc_x][loc_y]==TEAM2_UGV:
                    if delta_x<0: return RIGHT
                    elif delta_x>0: return LEFT
                    else: return STAY
        return UNDEFINED

    def find_enemy_flag_nearby(self, x, y, obs):
        for action in [UP, RIGHT, DOWN, LEFT]:
            next_x, next_y = self.next_position(x, y, action)
            if not self.check_out_of_bound(next_x, next_y, len(obs), len(obs[0])) and \
                obs[next_x][next_y]==TEAM2_FLAG:
                return action
        return UNDEFINED

    def check_dangerous(self, a_range, x, y, obs):
        for delta_x in range(-2*a_range, 2*a_range+1):
            for delta_y in range(-2*a_range, 2*a_range+1):
                loc_x, loc_y = x+delta_x, y+delta_y
                if delta_x**2 + delta_y**2 <= (a_range*2)**2 and \
                    not self.check_out_of_bound(loc_x, loc_y, len(obs), len(obs[0])) and \
                    obs[loc_x][loc_y]==TEAM2_UGV:
                    return True
        return False
