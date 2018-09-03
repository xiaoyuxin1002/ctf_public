"""

Roomba agents policy generator.

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

    def gen_action(self, agent_list, obs, free_map=None):
        if map is not None:
            self.map = free_map

        action_out = []

        for idx,agent in enumerate(agent_list):
            a = self.roomba(agent, idx, self.identity_list[idx], obs)
            action_out.append(a)

        return action_out

    def roomba(self, agent, index, identity, obs):
        if identity == Patrolling_Group:
            return self.patrol(agent, index, obs)
        else:
            return self.explore(agent, index, obs)

    def patrol(self, agent, index, obs):
        x, y = agent.get_loc()

        # if not along the boundary,
        # go down to approach the boundary
        if y < len(obs[0]) / 2 - 1:
            # if it is possible to go down, then go down
            action = DOWN
            next_x, next_y = self.next_position(x, y, action)
            if not self.check_obstacle(next_x, next_y, obs):
                self.prev_action_list[index] = action
                return action
            else:
                # if it is possible continue the previous direction, then continue
                action = self.prev_action_list[index]
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
                    for action in potential_actions:
                        next_x, next_y = self.next_position(x, y, action)
                        if not self.check_obstacle(next_x, next_y, obs):
                            self.prev_action_list[index] = action
                            return action
                    # TODO: need to solve the extreme case of being trapped in a concave shape

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
        return STAY

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

    def find_attacking_direction(self, a_range, x, y, obs):
        for delta_x in range(-2*a_range, 2*a_range+1):
            for delta_y in range(0, 2*a_range+1):
                loc_x, loc_y = x+delta_x, y+delta_y
                if delta_x**2 + delta_y**2 <= (a_range*2)**2 and \
                    not self.check_out_of_bound(loc_x, loc_y, len(obs), len(obs[0])) and \
                    obs[loc_x][loc_y]==TEAM2_UGV:
                    if delta_x<0: return RIGHT
                    elif delta_x>0: return LEFT
                    else: return STAY
        return UNDEFINED

        # x,y = agent.get_loc()
        # action = self.prev_action[index]
        #
        # self.count = self.count - 1
        #
        # # if cross the boundary, go back to avoid being killed
        # if (y >= len(obs[0])):
        #     action = 1
        #     self.prev_action[index] = action
        #     return action
        #
        # # increase randomness in actions for exploration
        # # and avoid the same pattern
        # if (self.count <= 0):
        #     self.count = 100
        #     action = np.random.randint(1, 5)
        #     self.prev_action[index] = action
        #     return action
        #
        # if (action == 1 or action == 3):
        #     # if entering a new horizontal line, exploring this line
        #     possible_actions = []
        #     for possible_action in [2, 4]:
        #         next_x, next_y = self.next_position(x, y, possible_action)
        #         if (not self.check_obstacle(next_x, next_y, obs)):
        #             possible_actions.append(possible_action)
        #     if (possible_actions != []):
        #         for i in range(3):
        #             action = np.random.choice(possible_actions)
        #             if (not self.within_attack_range(agent.a_range, x, y, obs)):
        #                 self.prev_action[index] = action
        #                 return action
        #         return 0
        #     else:
        #         # if being traped in a corner, turn back
        #         if (action == 1):
        #              action = 3
        #              self.prev_action[index] = action
        #              return action
        #         else:
        #              action = 1
        #              self.prev_action[index] = action
        #              return action
        # else:
        #     next_x, next_y = self.next_position(x, y, action)
        #     # continue exploring the current horizontal line if possible
        #     if (not self.check_obstacle(next_x, next_y, obs)):
        #         if (not self.within_attack_range(agent.a_range, x, y, obs)):
        #             return action
        #         return 0
        #     else:
        #         # otherwist, change to a new line
        #         possible_actions = []
        #         for possible_action in [1, 3]:
        #             next_x, next_y = self.next_position(x, y, possible_action)
        #             if (not self.check_obstacle(next_x, next_y, obs) and
        #                not self.within_attack_range(agent.a_range, x, y, obs)):
        #                 possible_actions.append(possible_action)
        #         if (possible_actions != []):
        #             for i in range(3):
        #                 action = np.random.choice(possible_actions)
        #                 if (not self.within_attack_range(agent.a_range, x, y, obs)):
        #                     self.prev_action[index] = action
        #                     return action
        #             return 0
        #         else:
        #             # if being traped in a corner, turn back
        #             if (action == 2):
        #                  action = 4
        #                  self.prev_action[index] = action
        #                  return action
        #             else:
        #                  action = 2
        #                  self.prev_action[index] = action
        #                  return action
