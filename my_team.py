# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MitalDefensive', second='MitalOffensiveRata', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}




class MitalDefensive(ReflexCaptureAgent):
    """Simple example agent: picks a random legal action (not STOP)"""
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.prev_food = []
        self.counter = 0

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.camping(game_state)

    def camping(self, game_state):
        #goes to the middle of the map and starts camping (jugador tactico) the zone
        x = (game_state.data.layout.width - 2) // 2
        if not self.red:
            x+=1
        self.camping_pos = []

        #get all the positions in the middle column that are not walls
        for i in range(1, game_state.data.layout.height - 1):
            if not game_state.has_wall(x, i):
                self.camping_pos.append((x, i))
        
        #remove the top and bottom positions to avoid camping near the borders (we will use this code to make the attacker go first on top or bottom)
        for i in range(len(self.camping_pos)):
            if len(self.camping_pos) > 2:
                self.camping_pos.remove(self.camping_pos[0])
                self.camping_pos.remove(self.camping_pos[-1])
            else:
                break
    
    def next_move(self, game_state):

        agent_moves = []
        possible_moves = game_state.get_legal_actions(self.index)
        
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        possible_moves.remove(Directions.STOP)

        #remove the reverse direction to avoid going back
        for i in range(0, len(possible_moves)-1):
            if reverse == possible_moves[i]:
                possible_moves.remove(reverse)

        #only keep moves that keep the agent in defense mode (not pacman)
        for i in range(len(possible_moves)):
            j = possible_moves[i]
            new_state = game_state.generate_successor(self.index, j)
            if not new_state.get_agent_state(self.index).is_pacman:
                agent_moves.append(j)
        
        #if no moves left, reset counter
        if len(agent_moves) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agent_moves.append(reverse)

        return agent_moves 

    def choose_action(self, game_state):
        #select next enemy to target
        position = game_state.get_agent_position(self.index)
        #if we have reached the target, reset it
        if position == self.target:
            self.target = None
        #set initial conditions
        min_dist = float("inf")
        enemies = []
        nearest_enemy = []


        #search for enemies at base   
        enemy_location_array = self.get_opponents(game_state)
        i = 0
        while i != len(enemy_location_array):
            enemy_position = enemy_location_array[i]
            enemy = game_state.get_agent_state(enemy_position)

            #check if the enemy is attacking and if its position is known
            if enemy.is_pacman and enemy.get_position() != None:
                enemy_position = enemy.get_position()
                enemies.append(enemy_position)
            i+=1

        #target the closest enemy in case of finding
        if len(enemies) > 0:
            for e in enemies:
                dist = self.get_maze_distance(e, position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy.append(e)
            self.target = nearest_enemy[-1]

        #if enemy not found, check if food was eaten
        else:
            if len(self.prev_food) > 0:
                #food was eaten
                if len(self.get_food_you_are_defending(game_state).as_list()) < len(self.prev_food):
                    eaten_food = set(self.prev_food) - set(self.get_food_you_are_defending(game_state).as_list())
                    self.target = eaten_food.pop()
        #update previous food list
        self.prev_food = self.get_food_you_are_defending(game_state).as_list()
        
        #if no target, start camping or defending food
        if self.target == None:
            if len(self.get_food_you_are_defending(game_state).as_list()) <= 4:
                vip_food = self.get_food_you_are_defending(game_state).as_list() + self.get_capsules_you_are_defending(game_state)
                self.target = random.choice(vip_food)
            else:
                self.target = random.choice(self.camping_pos)
        possible_move = self.next_move(game_state)
        best_move = []
        dist = []

        i=0
        
        #choose the best move according to the maze distance to the target      
        while i < len(possible_move):
            a = possible_move[i]
            next_state = game_state.generate_successor(self.index, a)
            new_pos = next_state.get_agent_position(self.index)
            best_move.append(a)
            dist.append(self.get_maze_distance(new_pos, self.target))
            i+=1

        best = min(dist)
        best_moves = [a for a, v in zip(best_move, dist) if v == best]
        the_chosen_one = random.choice(best_moves)
        return the_chosen_one  


class MitalOffensiveRata(ReflexCaptureAgent):
    """
    Offensive agent that only grabs the minimum food needed to win,
    then switches to defensive behavior.
    """
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.mode = 'offense'  # 'offense' or 'defense'
        self.prev_food = []
        self.counter = 0
        self.camping_pos = []
        
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start_pos = game_state.get_agent_position(self.index)
        self.setup_camping_positions(game_state)
        
    def setup_camping_positions(self, game_state):
        """Set up defensive camping positions on our side"""
        x = (game_state.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.camping_pos = []
        
        for i in range(1, game_state.data.layout.height - 1):
            if not game_state.has_wall(x, i):
                self.camping_pos.append((x, i))
        
        # Remove top and bottom positions
        for i in range(len(self.camping_pos)):
            if len(self.camping_pos) > 2:
                self.camping_pos.remove(self.camping_pos[0])
                self.camping_pos.remove(self.camping_pos[-1])
            else:
                break
    
    def should_switch_to_offense(self, game_state):
        """Check if we should switch to offense mode"""
        score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        
        # Switch to offense if we're losing or tied AND we're not carrying food
        return score <= 0 and my_state.num_carrying == 0
    
    def should_switch_to_defense(self, game_state):
        """Check if we should switch to defense mode"""
        score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        
        # Switch to defense if we're winning AND we've deposited food OR we're safely home
        return score > 0 and (my_state.num_carrying == 0 or not my_state.is_pacman)
    
    def get_food_needed(self, game_state):
        """Calculate how many food pellets we need to win"""
        score = self.get_score(game_state)
        my_state = game_state.get_agent_state(self.index)
        
        # We need enough to make score > 0
        if score <= 0:
            return abs(score) + 1 - my_state.num_carrying
        return 0
    
    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = game_state.get_agent_position(self.index)
        
        # Decide mode based on game state
        if self.should_switch_to_offense(game_state):
            self.mode = 'offense'
        elif self.should_switch_to_defense(game_state):
            self.mode = 'defense'
        
        # OFFENSE MODE: Get minimum food needed to win
        if self.mode == 'offense':
            food_needed = self.get_food_needed(game_state)
            
            # If we have enough food, return home
            if my_state.num_carrying >= food_needed and my_state.num_carrying > 0:
                return self.return_home(game_state)
            
            # Otherwise, go get food
            return self.get_food_action(game_state)
        
        # DEFENSE MODE: Use defensive strategy from MitalDefensive
        else:
            return self.defend(game_state)
    
    def return_home(self, game_state):
        """Navigate back to our side to deposit food"""
        my_pos = game_state.get_agent_position(self.index)
        
        # Find closest border position
        border_x = game_state.data.layout.width // 2
        if not self.red:
            border_x -= 1
        
        border_positions = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(border_x, y):
                border_positions.append((border_x, y))
        
        if not border_positions:
            return random.choice(game_state.get_legal_actions(self.index))
        
        # Find closest border position
        min_dist = float('inf')
        best_border = None
        for border_pos in border_positions:
            dist = self.get_maze_distance(my_pos, border_pos)
            if dist < min_dist:
                min_dist = dist
                best_border = border_pos
        
        # Move towards closest border
        actions = game_state.get_legal_actions(self.index)
        actions.remove(Directions.STOP)
        
        best_action = None
        best_dist = float('inf')
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, best_border)
            
            if dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def get_food_action(self, game_state):
        """Get action to collect food"""
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        
        if not food_list:
            return self.return_home(game_state)
        
        # Find closest food, avoiding ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        # Filter out dangerous food near ghosts
        safe_food = []
        for food in food_list:
            is_safe = True
            for ghost in ghosts:
                if ghost.scared_timer == 0:
                    ghost_dist = self.get_maze_distance(food, ghost.get_position())
                    if ghost_dist < 3:
                        is_safe = False
                        break
            if is_safe:
                safe_food.append(food)
        
        # If no safe food, use all food
        target_food = safe_food if safe_food else food_list
        
        # Find closest food
        min_dist = float('inf')
        closest_food = None
        for food in target_food:
            dist = self.get_maze_distance(my_pos, food)
            if dist < min_dist:
                min_dist = dist
                closest_food = food
        
        # Move towards closest food
        actions = game_state.get_legal_actions(self.index)
        actions.remove(Directions.STOP)
        
        best_action = None
        best_dist = float('inf')
        
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, closest_food)
            
            # Avoid ghosts
            too_close_to_ghost = False
            for ghost in ghosts:
                if ghost.scared_timer == 0:
                    ghost_dist = self.get_maze_distance(new_pos, ghost.get_position())
                    if ghost_dist <= 1:
                        too_close_to_ghost = True
                        break
            
            if not too_close_to_ghost and dist < best_dist:
                best_dist = dist
                best_action = action
        
        return best_action if best_action else random.choice(actions)
    
    def defend(self, game_state):
        """Defensive behavior - same as MitalDefensive"""
        position = game_state.get_agent_position(self.index)
        
        if position == self.target:
            self.target = None
        
        invaders = []
        nearest_invader = []
        min_distance = float("inf")
        
        opponents_positions = self.get_opponents(game_state)
        for opponent_pos in opponents_positions:
            opponent = game_state.get_agent_state(opponent_pos)
            if opponent.is_pacman and opponent.get_position() is not None:
                opponent_position = opponent.get_position()
                invaders.append(opponent_position)
        
        if len(invaders) > 0:
            for opp_position in invaders:
                dist = self.get_maze_distance(opp_position, position)
                if dist < min_distance:
                    min_distance = dist
                    nearest_invader.append(opp_position)
            self.target = nearest_invader[-1]
        else:
            if len(self.prev_food) > 0:
                if len(self.get_food_you_are_defending(game_state).as_list()) < len(self.prev_food):
                    eaten_food = set(self.prev_food) - set(self.get_food_you_are_defending(game_state).as_list())
                    self.target = eaten_food.pop()
        
        self.prev_food = self.get_food_you_are_defending(game_state).as_list()
        
        if self.target is None:
            if len(self.get_food_you_are_defending(game_state).as_list()) <= 4:
                vip_food = self.get_food_you_are_defending(game_state).as_list() + \
                          self.get_capsules_you_are_defending(game_state)
                self.target = random.choice(vip_food)
            else:
                self.target = random.choice(self.camping_pos)
        
        possible_moves = self.get_defensive_moves(game_state)
        best_moves = []
        distances = []
        
        for action in possible_moves:
            next_state = game_state.generate_successor(self.index, action)
            new_pos = next_state.get_agent_position(self.index)
            best_moves.append(action)
            distances.append(self.get_maze_distance(new_pos, self.target))
        
        best = min(distances)
        best_actions = [a for a, v in zip(best_moves, distances) if v == best]
        return random.choice(best_actions)
    
    def get_defensive_moves(self, game_state):
        """Get legal defensive moves"""
        agent_moves = []
        possible_moves = game_state.get_legal_actions(self.index)
        
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        possible_moves.remove(Directions.STOP)
        
        for i in range(0, len(possible_moves) - 1):
            if reverse == possible_moves[i]:
                possible_moves.remove(reverse)
        
        for move in possible_moves:
            new_state = game_state.generate_successor(self.index, move)
            if not new_state.get_agent_state(self.index).is_pacman:
                agent_moves.append(move)
        
        if len(agent_moves) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
            
        if self.counter > 4 or self.counter == 0:
            agent_moves.append(reverse)
        
        return agent_moves