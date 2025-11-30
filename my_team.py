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

# Our team is composed by a defensive agent that stays in the border and an offensive one that 
# gets enough food to win and then switches to defenseas the other one. 

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


class MitalDefensive(ReflexCaptureAgent):
    """
    Mital Defensive agent:
    
    This agent has the goal of staying in the border between both sides. It will 
    try to block the entrance of the enemies to our side, giving priority to chasing
    enemies  that manage to get to the side. It will also constantly keep track of the 
    previous food state to detect if there is missing food. In that case it will target 
    that dot of food and it will try to hunt down the enemy that got past the defenses.
    """
    
    #We initilaize the agent
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.prev_food = []
        self.counter = 0

    #We set the initial state and camping positions
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.camping(game_state)

    #We compute the camping positions, wich are the ones located in the middle of the map (because we think that most of the times enemies will try to cross there)
    def camping(self, game_state):
        #goes to the middle of the map and starts camping (jugador tactico) the zone
        width = game_state.data.layout.width
        x = (width // 2) - 1
        
        #if we are blue, we add one to x so we are in our side of the map (the right one)
        if not self.red:
            x+=1
        self.camping_pos = []

        #get all the positions in the middle column that are not walls so we can camp there
        h = game_state.data.layout.height
        i = 1
        while i < h - 1:
            if game_state.has_wall(x, i) == False:
                self.camping_pos.append((x, i)) 
            i += 1
            
        #remove the top and bottom positions to avoid camping near the borders (we maybe will use this code to make the attacker go first on top or bottom)
        for i in range(len(self.camping_pos)):
            if len(self.camping_pos) > 2:
                self.camping_pos.remove(self.camping_pos[0])
                self.camping_pos.remove(self.camping_pos[-1])
            else:
                break
    
    #we get the possible defensive moves, avoiding stopping. These moves will be then checked so we don't go into pacman mode
    def next_move(self, game_state):

        agent_moves = []
        #get legal moves
        possible_moves = game_state.get_legal_actions(self.index)
        
        #get reverse direction and remove stop option
        reverse = Directions.REVERSE[
            game_state.get_agent_state(self.index).configuration.direction
        ]
        
        #we remove stop from possible moves
        possible_moves.remove(Directions.STOP)

        #remove the reverse direction to avoid going back too much (we are interested in staying in the border to capture the most enemies)
        for i in range(0, len(possible_moves)-1):
            if reverse == possible_moves[i]:
                possible_moves.remove(reverse)

        #only keep moves that keep the agent in defense mode (not pacman)
        for i in range(len(possible_moves)):    
            j = possible_moves[i]
            
            #we generate the succesor state to check if we would be pacman or not
            new_state = game_state.generate_successor(self.index, j)
            
            #if we dont convert to pacman, we keep the move, else we dont
            if not new_state.get_agent_state(self.index).is_pacman:
                agent_moves.append(j)
        
        #if no moves left, reset counter
        if len(agent_moves) == 0:
            self.counter = 0
        else:
            # increment counter
            self.counter += 1
        
        #if we have been avoiding reverse for too long, allow it again. (we want to be on the border but sometimes we need to go back so we dont get stuck)
        if self.counter > 4 or self.counter == 0:
            agent_moves.append(reverse)

        # we finally return the possible moves (for camping the area)
        return agent_moves 

    #Previously we computed the moves for defense, now we will either chose to defend (camping in the border), chase enemies (if they are close to us) or go 
    # to the missing food (which is the case in which an enemy got past our defenses and we dind't notice it in time)
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
        
        #check all enemy positions
        for enemy_position in enemy_location_array:
            
            #we get the enemy state
            enemy = game_state.get_agent_state(enemy_position)
        
            #get the enemy position 
            pos = enemy.get_position()
            
            if enemy.is_pacman is True:
                if pos is not None:
                    enemies.append(pos)
                        
        #target the closest enemy in case of finding
        if enemies:
            
            #check all enemies found
            for e in enemies:
                #compute distance to enemy
                dist = self.get_maze_distance(e, position)
                
                #if the distance is smaller than the previous minimum, update it and set the enemy as target
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy.append(e)
                    
            #set the target to the nearest enemy (the one in the last position of the list)
            self.target = nearest_enemy[-1]

        #if enemy not found, check if food was eaten
        else:
            #get the current food list
            current_food = self.get_food_you_are_defending(game_state).as_list()
            
            #if the if below is true, we had food before
            if self.prev_food:
                
                #food was eaten if the if below is true
                if len(current_food) < len(self.prev_food):
                    
                    #compute which food was eaten
                    eaten_food = set(self.prev_food) - set(current_food)
                    
                    #target that food and go there (it means an enemy ate it)
                    self.target = eaten_food.pop()
                    
        #update previous food list
        self.prev_food = self.get_food_you_are_defending(game_state).as_list()
        
        #if no target, start camping or defending food. We don't need to check for enemies or eaten food
        if self.target is None:
            
            #check if we have few food left to defend (we consider few food when there are 4 or less dots)
            if len(current_food) <= 4:
                few_food_left = True
            else:
                few_food_left = False
            
            #if we are low on food, we give priority to defending food and capsules
            if few_food_left:
                vip_food = current_food + self.get_capsules_you_are_defending(game_state)
                
                #set target to a random food or capsule to defend
                self.target = random.choice(vip_food)
            else:
                #if we are not low on food, we just camp in the middle of the map as always
                self.target = random.choice(self.camping_pos)
        
        #the possible moves are computed using the function before        
        possible_move = self.next_move(game_state)
        best_move = []
        dist = []

        #get the best move to the target using the maze distance (as in the example code)
        for a in possible_move:
            #generate the successor state and get the new position
            next_state = game_state.generate_successor(self.index, a)
            new_pos = next_state.get_agent_position(self.index)
            
            #append the action to the list of best moves
            best_move.append(a)
            
            #compute the maze distance to the target
            distance_value = self.get_maze_distance(new_pos, self.target)
            dist.append(distance_value)

        #get the minimum distance found
        best = min(dist)

        #initialize best moves
        best_moves = []
        k = 0
        
        #iterate over the distances to find all the best moves (in case of ties)
        while k < len(dist):
            if dist[k] == best:
                best_moves.append(best_move[k])
            k += 1
        
        #ramdomly choose one of the best moves in case of ties
        the_chosen_one = random.choice(best_moves)
        
        #then return it
        return the_chosen_one  


class MitalOffensiveRata(MitalDefensive):

    """
    Mital Offensive agent:
    
    This agent has two modes: OFFENSE and DEFENSE. The idea is that at first, as the game is tied, it will go 
    to the enemy side and collect exactly enough food to produce a winning score. Once it has collected enough 
    food, it will go back to it's part of the map to deposit the food and it will change to DEFENSE mode, behaving exactly
    like MitalDefensive. The OFFENSE mode uses a reflex aproach, chasing food and avoiding ghosts.
    
    If further in the game it detects that we are no longer winning (maybe because the enemy scored), it will switch back to OFFENSE 
    mode and so on.
    """

    def __init__(self, index):
        super().__init__(index)
        self.mode = "offense" #"offense" or "defense"
        
        #preposition
        self.first_time_positioning = True
        self.positioning_counter = 0
        self.POSITIONING_DURATION = 45
        
        #tactilcal dots for both teams (it is a dot of food that may catch our agent if not careful)
        #thats why we will go there at the beguining just in case someone wants to rush that place
        
        # self.red_position = (10, 3)
        self.red_position = (16, 2)

        # self.blue_position = (20, 12)
        self.blue_position = (15, 13)

        self.tactical_target = None


    #we compute how many dots we have to eat to have a winning score
    def food_needed_to_win(self, game_state):
        score = game_state.get_score()

        #this buffer allows us to choose the advantage we want to have when winning
        buffer = 2
        
        if self.red:
            #Red must score > 0
            needed = buffer - score
        else:
            #Blue must score < 0
            needed = score + buffer

        #if needed is negative, we don't need any food to win
        if needed < 0:
            needed = 0
        return needed

    #Reflex aproach for the offensive mode of the agent
    def choose_offensive_action(self, game_state):

        #prepositioning just the first time:
        if self.first_time_positioning:
            # we go to the tactical food dot depending on the team
            if self.tactical_target is None:
                #red
                if self.red:
                    self.tactical_target = self.red_position
                #blue
                else:
                    self.tactical_target = self.blue_position

            #get legal actions
            actions = game_state.get_legal_actions(self.index)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)

            #go to the tactical target
            position = game_state.get_agent_position(self.index)
            best_action = None
            min_distance = float("inf")

            for a in actions:
                successor = game_state.generate_successor(self.index, a)
                new_pos = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(new_pos, self.tactical_target)
                if dist < min_distance:
                    min_distance = dist
                    best_action = a

            #increment positioning counter 
            self.positioning_counter += 1
            
            #after some time, we stop pre-positioning
            if self.positioning_counter >= self.POSITIONING_DURATION:
                self.first_time_positioning = False  # Done pre-positioning
            
            #return the best action to go to the tactical dot
            return best_action

        #continue as normal

        #We get the legal actions
        actions = game_state.get_legal_actions(self.index)

        #as in the defensive, we avoid stopping
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        #basic info about our agent
        my_state = game_state.get_agent_state(self.index)
        carried = my_state.num_carrying
        position = game_state.get_agent_position(self.index)

        #get the food on the enemy side
        enemy_food = self.get_food(game_state).as_list()

        #get visible enemy defenders (ghosts) so that we can avoid them
        opponents = self.get_opponents(game_state)
        visible_ghosts = []
        
        #iterate over all oponents to find the visible ghosts
        for opp in opponents:
            enemy = game_state.get_agent_state(opp)
            #we only care about visible ghosts (not pacman)
            if not enemy.is_pacman and enemy.get_position() is not None:
                
                #we add it to the list of visible ghosts
                visible_ghosts.append(enemy.get_position())

        #compute how much food is needed to win (will check it always)
        needed = self.food_needed_to_win(game_state)

        #if we already have enough food, ew go home
        if carried >= needed and needed > 0:
            return self.go_home(game_state, actions)

        #The code below is an attempt to avoid getting stuck going for the same food
        #for too long when blocked by ghosts. It doesn't really quite work but we will leave it here

        #we initialize the target if it doesn't exist
        try:
            self.current_target
        except AttributeError:
            self.current_target = None
            self.target_counter = 0
            self.blocked_target = {}

        #increment counter for blocked targets
        for target in list(self.blocked_target.keys()):
            self.blocked_target[target] -= 1
            if self.blocked_target[target] <= 0:
                del self.blocked_target[target]

        #if we have a target, we check if we have been going for it for too long
        if enemy_food:
            
            #compute available food (not blocked)
            available_food = []
            for f in enemy_food:
                if f not in self.blocked_target:
                    available_food.append(f)

            if not available_food:  
                available_food = enemy_food #if all are blocked

            #we compute an array of distances to all food
            distances = []
            for f in enemy_food:
                dist = self.get_maze_distance(position, f)
                distances.append((dist, f))
            
            #we sort the distances to get the closest food
            distances.sort()
            
            #we choose the closest food as target
            chosen_target = distances[0][1]

            #if the agent is stuck, increment a counter
            if self.current_target == chosen_target:
                self.target_counter += 1
            
            #if we have been going for the same target for too long, we change target
            else:
                self.current_target = chosen_target
                self.target_counter = 0
                
            #The maximum counter value before changing target
            MAX_COUNTER = 5
            BLOCK_DURATION = 20
            
            #if we exceed the maximum counter, we change target
            if self.target_counter > MAX_COUNTER:
                self.blocked_target[self.current_target] = BLOCK_DURATION
                other_food = []
                #we create a list of other food (not the current target)
                for f in enemy_food:
                    if f != self.current_target:
                        other_food.append(f)
                
                #if there is another food, we change target to it
                if other_food:
                    
                    #same as before, we compute distances to other food
                    distances2 = []
                    for f in other_food:
                        dist = self.get_maze_distance(position, f)
                        distances2.append((dist, f))

                    #sort it 
                    distances2.sort()
                    
                    #set the new target
                    chosen_target = distances2[0][1]
                
                #finally we update the current target and reset the counter
                self.current_target = chosen_target
                self.target_counter = 0
        
        #in case there is no food, we set the target to None
        else:
            chosen_target = None

        #we iterate over all actions to see what is best (this is the reflex part)
        scores = []
        for a in actions:
            
            #generate the successor state and get the new position
            successor = game_state.generate_successor(self.index, a)
            new_pos = successor.get_agent_position(self.index)

            #initialize reflex value
            value = 0

            food_dist = float("inf")
                
            #we want food that is closer to us
            if enemy_food:
                #so we compute the distance to the closest food
                for f in enemy_food:
                    temp_food_dist = self.get_maze_distance(new_pos, f)
                    
                    #we compute the minimum distance to food
                    if temp_food_dist < food_dist:
                        food_dist = temp_food_dist
                        
                #we increase the value the closer the food is
                value += 10.0 / (1 + food_dist)

            #we want to avoid ghosts, so we decrease the value if we are close to them
            for g in visible_ghosts:
                
                #compute distance to ghost
                d = self.get_maze_distance(new_pos, g)
                
                #if we are too close to a ghost, heavily decrease the value
                if d < 4:
                    value -= (5.0 / (1 + d))

            #append the value and action to the scores list
            scores.append((value, a))

        #compute the best value
        best_value = -float("inf")
        
        #iterate over all scores to find the best value
        for pair in scores:
            
            #get the value from the pair
            v = pair[0]
            if v > best_value:
                best_value = v

        #compute the best actions
        best_actions = []
        for v, a in scores:
            
            #if the value is the best, we add the action to the list
            if v == best_value:
                best_actions.append(a)

        #if there is a tie, we randomly choose one of the best actions
        return random.choice(best_actions)

    #this function makes the agent go back to its side of the map
    def go_home(self, game_state, actions):

        #we get the current position
        position = game_state.get_agent_position(self.index)

        #we will choose the action that gets us closer to the camping positions (which are in our side of the map)
        best_distance = float("inf") #because we want to minimize distance
        chosen_action = None

        #iterate over all actions
        for a in actions:
            
            #generate the successor state and get the new position (same as before)
            successor = game_state.generate_successor(self.index, a)
            new_pos = successor.get_agent_position(self.index)

            #we compute the nearest distance to our camping positions
            d = float("inf")
            for c in self.camping_pos:
                dist = self.get_maze_distance(new_pos, c)
                if dist < d:
                    d = dist
                    
            #we choose the action that minimizes the distance to our side
            if d < best_distance:
                best_distance = d
                chosen_action = a

        #finally we return the chosen action
        return chosen_action

    #this is the "main" function of the offensive agent and is responsible for switching modes
    def choose_action(self, game_state):

        #the global score will decide in which mode we are
        score = game_state.get_score()

        #if we are red, positive score means we are winning, else negative score means we are winning
        if self.red:
            winning = (score > 0)
        else:
            winning = (score < 0)

        #if we are winning, we switch to defense mode, else we stay in offense mode
        if winning:
            self.mode = "defense"
        else:
            self.mode = "offense"

        #if we are in defense mode, we use the defensive choose action (called to the suoerclass MitalDefensive)
        if self.mode == "defense":
            return super().choose_action(game_state)

        #If we are in offense mode, we use the offensive choose action
        return self.choose_offensive_action(game_state)
