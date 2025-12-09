# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import random

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MitalHybrid', second='MitalHybrid', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    
    Both agents are MitalHybrid that switch roles when one dies.
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


class MitalDefensive(ReflexCaptureAgent):
    """
    The Mital Defensive agent camps near the border and chases invaders that manage to get through.
    It keeps track of the food in the previous state to check if any food has been eaten. 
    When the enemy has a power pellet, it maintains a safe distance while trying to shadow them.
    """
    
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        #set initial target to None, previous food list to empty, and counter to 0
        self.target = None
        self.prev_food = []
        self.counter = 0
        self.camping_pos = []

    #set the initial state and camping positions
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.camping(game_state)

    #this function sets up camping positions near the border
    def camping(self, game_state):
        #compute x coordinate for camping positions
        width = game_state.data.layout.width
        x = (width // 2) - 1

        #to get the middle correctly for blue team
        if not self.red:
            x += 1
        self.camping_pos = []

        h = game_state.data.layout.height
        for i in range(1, h - 1):
            if not game_state.has_wall(x, i):
                self.camping_pos.append((x, i))

        #remove extreme positions to avoid being stuck
        for i in range(len(self.camping_pos)):
            if len(self.camping_pos) > 2:
                self.camping_pos.remove(self.camping_pos[0])
                self.camping_pos.remove(self.camping_pos[-1])
            else:
                break

    #we get the next defensive moves avoiding going back and forth by deleting reverse move
    def next_move(self, game_state):
        agent_moves = []
        possible_moves = game_state.get_legal_actions(self.index)

        #determine reverse direction
        reverse = Directions.REVERSE[
            game_state.get_agent_state(self.index).configuration.direction
        ]

        #avoid stopping
        if Directions.STOP in possible_moves:
            possible_moves.remove(Directions.STOP)

        #remove reverse move to avoid oscillation
        for i in range(len(possible_moves) - 1, -1, -1):
            if reverse == possible_moves[i]:
                possible_moves.remove(reverse)
                break

        #only consider moves that keep us on defense (not pacman)
        for move in possible_moves:
            new_state = game_state.generate_successor(self.index, move)
            if not new_state.get_agent_state(self.index).is_pacman:
                agent_moves.append(move)

        #if no valid moves left, allow reverse move after some time
        if len(agent_moves) == 0:
            self.counter = 0
        else:
            self.counter += 1

        #if we have been avoiding reverse for too long, allow it again
        if self.counter > 4 or self.counter == 0:
            agent_moves.append(reverse)

        return agent_moves

    #check if any enemy invader has eaten our power pellet (this is new since the first test contest in class)
    def is_enemy_powered(self, game_state):
        #get the enemies in a compact form
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        #check if any enemy is pacman and we are scared
        for enemy in enemies:
            if enemy.is_pacman and enemy.get_position() is not None:
                # check if we are scared
                my_state = game_state.get_agent_state(self.index)
                if my_state.scared_timer > 0:
                    return True, enemy.get_position(), my_state.scared_timer
        
        #if no powered enemy found
        return False, None, 0

    #determine safe shadowing position to keep a distance from powered enemies. (This is also new from first test contest, as in the past version we didn't consider power pellets)
    def get_safe_shadowing_position(self, my_pos, enemy_pos, scared_time):
        
        #safe distance (3-4 cells) but close enough to chase when power runs out.
        safe_distance = 4 if scared_time > 5 else 3
        
        dx = my_pos[0] - enemy_pos[0]
        dy = my_pos[1] - enemy_pos[1]
        
        current_dist = abs(dx) + abs(dy)
        
        #decide what to do based on current distance
        if current_dist < safe_distance:
            return "RETREAT"
        elif current_dist > safe_distance + 2:
            return "APPROACH"
        else:
            return "MAINTAIN"

    #main function of Mital Defensive agent, that consists on taking into account everything defined before
    #and choosing the best action acording to the results of previous functions
    def choose_action(self, game_state):
        #our current position
        position = game_state.get_agent_position(self.index)
        
        #see if any enemy is powered
        is_powered, powered_enemy_pos, scared_time = self.is_enemy_powered(game_state)
        
        if is_powered and powered_enemy_pos is not None:
            #the enemy has a power pellet, so we need to be careful
            my_pos = position
            shadow_strategy = self.get_safe_shadowing_position(my_pos, powered_enemy_pos, scared_time)
            
            possible_moves = self.next_move(game_state)
            best_action = None
            
            #decide action based on shadowing strategy
            #RETREAT
            if shadow_strategy == "RETREAT":
                best_dist = 0
                for action in possible_moves:
                    successor = game_state.generate_successor(self.index, action)
                    new_pos = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(new_pos, powered_enemy_pos)
                    
                    #we want to maximize distance from powered enemy
                    if dist > best_dist:
                        best_dist = dist
                        best_action = action
                    
            #APPROACH
            elif shadow_strategy == "APPROACH":
                best_dist = float('inf')
                for action in possible_moves:
                    successor = game_state.generate_successor(self.index, action)
                    new_pos = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(new_pos, powered_enemy_pos)
                    
                    #we want to minimize distance but keep a safe margin
                    if dist < best_dist and dist >= 3:
                        best_dist = dist
                        best_action = action
            else:
                #MAINTAIN
                target_dist = 3
                best_diff = float('inf')
                for action in possible_moves:
                    successor = game_state.generate_successor(self.index, action)
                    new_pos = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(new_pos, powered_enemy_pos)
                    diff = abs(dist - target_dist)
                    
                    #we want to keep distance close to target_dist
                    if diff < best_diff:
                        best_diff = diff
                        best_action = action
                        
            #execute the best action found
            if best_action:
                return best_action
        
        #normal defensive behavior when no enemy is powered
        if position == self.target:
            self.target = None

        min_dist = float("inf")
        enemies = []
        nearest_enemy = []

        enemy_indices = self.get_opponents(game_state)

        #gather positions of enemy pacman invaders
        for enemy_idx in enemy_indices:
            enemy = game_state.get_agent_state(enemy_idx)
            pos = enemy.get_position()

            #we only care about pacman enemies
            if enemy.is_pacman and pos is not None:
                enemies.append(pos)

        #find nearest enemy invader
        if enemies:
            for e in enemies:
                dist = self.get_maze_distance(e, position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy.append(e)
            
            #set target to nearest enemy
            self.target = nearest_enemy[-1]
        else:
            #no enemies detected, check if any food has been eaten
            current_food = self.get_food_you_are_defending(game_state).as_list()

            if self.prev_food:
                if len(current_food) < len(self.prev_food):
                    eaten_food = set(self.prev_food) - set(current_food)
                    self.target = eaten_food.pop()

        self.prev_food = self.get_food_you_are_defending(game_state).as_list()

        #if no target yet, go to camping position
        if self.target is None:
            current_food = self.get_food_you_are_defending(game_state).as_list()
            
            #if little food left, prioritize defending food and capsules
            if len(current_food) <= 4:
                vip_food = current_food + self.get_capsules_you_are_defending(game_state)
                if vip_food:
                    self.target = random.choice(vip_food)
                else:
                    self.target = random.choice(self.camping_pos)
            else:
                self.target = random.choice(self.camping_pos)

        possible_moves = self.next_move(game_state)
        best_moves = []
        distances = []

        #evaluate distances for possible moves
        for action in possible_moves:
            next_state = game_state.generate_successor(self.index, action)
            new_pos = next_state.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, self.target)
            distances.append(dist)

        #find moves that minimize distance to target
        min_dist = min(distances)
        for i, dist in enumerate(distances):
            if dist == min_dist:
                best_moves.append(possible_moves[i])

        #we finally choose one of the best moves randomly
        return random.choice(best_moves)


class MitalOffensive(ReflexCaptureAgent):
    """
    We modified the original Mital Offensive agent to include power pellet strategy and better enemy awareness.
    1. The agent now prioritizes power pellets when enemies are nearby.
    2. After eating a power pellet, the agent ignores scared ghosts and focuses on maximizing food collection.
    3. The agent returns home more strategically to secure points (it stays at the enemy border when it returns, 
       not fully crossing and thus not getting the points, but with the stuck detection it switches roles and we 
       manage to get the points (stuck detection explained after)).
    """

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        #set initial target to None, previous food list to empty, and counter to 0, as well as camping positions and start position (to check for death)
        self.target = None
        self.prev_food = []
        self.counter = 0
        self.camping_pos = []
        self.start_pos = None
        
        #also track power pellet state
        self.power_pellet_target = None
        self.power_mode_active = False
        self.power_mode_timer = 0

    #set initial state and camping positions (same as before)
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start_pos = game_state.get_agent_position(self.index)
        self.setup_camping_positions(game_state)

    #same fuction as before to setup camping positions, this won't be necessary because we changed our strategy to focus on offense only. 
    #in the first idea that we had we wanted to win by only 1 point so that when we had the advantage we could camp near the border and 
    # avoid risks, but finally we decided to go full offense and switch roles with the teammate when stuck or when one dies. Creating the 
    #hybrid agent explained later.
    def setup_camping_positions(self, game_state):
        x = (game_state.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.camping_pos = []

        for i in range(1, game_state.data.layout.height - 1):
            if not game_state.has_wall(x, i):
                self.camping_pos.append((x, i))

        for i in range(len(self.camping_pos)):
            if len(self.camping_pos) > 2:
                self.camping_pos.remove(self.camping_pos[0])
                self.camping_pos.remove(self.camping_pos[-1])
            else:
                break

    #evaluate threat level from enemy ghosts, we will return several useful metrics
    def evaluate_threat_level(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        active_ghosts = []
        scared_ghosts = []
        
        #separate active and scared ghosts
        for enemy in enemies:
            if not enemy.is_pacman and enemy.get_position() is not None:
                if enemy.scared_timer > 0:
                    scared_ghosts.append((enemy, enemy.scared_timer))
                else:
                    active_ghosts.append(enemy)
        
        closest_ghost_dist = float('inf')
        nearby_count = 0
        
        #evaluate distances to active ghosts
        for ghost in active_ghosts:
            dist = self.get_maze_distance(my_pos, ghost.get_position())
            if dist < closest_ghost_dist:
                closest_ghost_dist = dist
            if dist <= 5:
                nearby_count += 1
        
        #return metrics
        return closest_ghost_dist, nearby_count, len(scared_ghosts) > 0, scared_ghosts

    #this functoin decides if we should go for a power pellet based on threat level
    def should_get_power_pellet(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        
        #if already in power mode, no need to get another pellet
        if self.power_mode_active:
            return None
        
        closest_ghost_dist, nearby_count, _, _ = self.evaluate_threat_level(game_state)
        
        capsules = self.get_capsules(game_state)
        
        #no capsules available
        if not capsules:
            return None
        
        #decide if we should get a power pellet based on threat level
        if closest_ghost_dist <= 5:
            closest_capsule = min(capsules, key=lambda cap: self.get_maze_distance(my_pos, cap))
            return closest_capsule
        
        if nearby_count >= 2 and closest_ghost_dist <= 7:
            closest_capsule = min(capsules, key=lambda cap: self.get_maze_distance(my_pos, cap))
            return closest_capsule
        
        return None

    #update power mode status based on scared ghosts
    def update_power_mode_status(self, game_state):
        _, _, has_scared, scared_ghosts = self.evaluate_threat_level(game_state)
        
        #if there are scared ghosts, enter power mode
        if has_scared and scared_ghosts:
            max_scared_time = max(timer for _, timer in scared_ghosts)
            self.power_mode_active = True
            self.power_mode_timer = max_scared_time
        
        # if already in power mode, decrement timer
        elif self.power_mode_timer > 0:
            self.power_mode_timer -= 1
            if self.power_mode_timer <= 0:
                self.power_mode_active = False
        
        # no scared ghosts, exit power mode
        else:
            self.power_mode_active = False

    #decide if we should return home based on food carried
    def should_return_home(self, game_state):
        """        
        Returns home if:
        - Carrying 8+ pellets normally
        - Carrying 12+ pellets if enemies are scared
        - Carrying 5+ pellets if active ghost is very close
        - Carrying any food and time is running out (< 100 moves left)
        - Carrying 3+ pellets and deep in enemy territory with ghost nearby
        """
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        
        if carrying == 0:
            return False
        
        #time running out
        if game_state.data.timeleft < 100:
            return True
        
        #evaluate threat level
        my_pos = game_state.get_agent_position(self.index)
        closest_ghost_dist, _, has_scared, _ = self.evaluate_threat_level(game_state)
        
        #compute distance from home border to see if we are far in enemy territory
        border_x = game_state.data.layout.width // 2
        if not self.red:
            border_x -= 1
        
        distance_from_home = abs(my_pos[0] - border_x)
        
        #active ghost very close
        if closest_ghost_dist <= 3 and carrying >= 5:
            return True
        
        #deep in enemy territory with ghost nearby
        if distance_from_home >= 5 and carrying >= 3 and closest_ghost_dist <= 6:
            return True
        
        #decide based on power mode and scared status
        if self.power_mode_active and self.power_mode_timer > 10:
            #with active power mode: collect up to 12 pellets
            return carrying >= 12
        elif has_scared:
            #with scared ghosts: collect up to 10 pellets
            return carrying >= 10
        else:
            #the normal case
            return carrying >= 8

    #function to return home given the return of should_return_home
    def return_home(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        #see if we are already home
        if not my_state.is_pacman:
            return None
        
        #compute border x coordinate for both teams
        border_x = game_state.data.layout.width // 2
        if not self.red:
            border_x += 1
        
        #get all possible border positions
        border_positions = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(border_x, y):
                border_positions.append((border_x, y))
        
        #if no border positions found, just pick any legal action
        if not border_positions:
            actions = game_state.get_legal_actions(self.index)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            return random.choice(actions) if actions else Directions.STOP
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        
        best_border = None
        best_score = float('-inf')
                
        #find the best border position to head towards
        for border_pos in border_positions:
            dist_to_border = self.get_maze_distance(my_pos, border_pos)
            
            #compute distance to nearest active ghost
            min_ghost_dist = float('inf')
            if active_ghosts:
                for ghost in active_ghosts:
                    ghost_dist = self.get_maze_distance(border_pos, ghost.get_position())
                    if ghost_dist < min_ghost_dist:
                        min_ghost_dist = ghost_dist
            
            #we want to maximize distance from ghosts while minimizing distance to border
            score = min_ghost_dist * 2 - dist_to_border
            
            #select the border position with the best score
            if score > best_score:
                best_score = score
                best_border = border_pos
        
        #move to the new target (best_border)
        return self.move_towards_target(game_state, best_border, avoid_ghosts=True)

    #definition of how we move towards a target with ghost avoidance
    def move_towards_target(self, game_state, target, avoid_ghosts=True):
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)
        
        #we dont want to stop
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        
        best_action = None
        best_dist = float('inf')
        
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        
        #evaluate each action
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(new_pos, target)
            
            #check for ghost proximity if we want to avoid them
            too_dangerous = False
            if avoid_ghosts:
                for ghost in active_ghosts:
                    ghost_dist = self.get_maze_distance(new_pos, ghost.get_position())
                    if ghost_dist <= 1:
                        too_dangerous = True
                        break
            
            #if there are no dangers, get this action
            if not too_dangerous and dist < best_dist:
                best_dist = dist
                best_action = action
        
        #if no safe action found, just pick a random legal action
        return best_action if best_action else random.choice(actions)

    #main offensive action function that will choose the best action based on all previous functions (same structure as before but calling the new functions)
    def get_offensive_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        #update power mode status
        self.update_power_mode_status(game_state)
        
        closest_ghost_dist, nearby_count, has_scared, scared_ghosts = self.evaluate_threat_level(game_state)
        
        #decide if we should return home
        if self.should_return_home(game_state):
            return self.return_home(game_state)
        
        #if in power mode, focus on food collection
        if self.power_mode_active and self.power_mode_timer > 5:
            food_list = self.get_food(game_state).as_list()
            
            #if there is no food left, return home
            if not food_list:
                return self.return_home(game_state)
            
            #go for the closest food without avoiding ghosts (we are powered)
            closest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))
            return self.move_towards_target(game_state, closest_food, avoid_ghosts=False)
        
        #if power mode is about to end and we are carrying food, return home
        elif self.power_mode_active and self.power_mode_timer <= 5 and my_state.num_carrying > 0:
            return self.return_home(game_state)
        
        #decide if we should get a power pellet
        capsule_target = self.should_get_power_pellet(game_state)
        
        #if so, head towards it
        if capsule_target:
            return self.move_towards_target(game_state, capsule_target, avoid_ghosts=True)
        
        food_list = self.get_food(game_state).as_list()
        
        #if no food left, return home
        if not food_list:
            return self.return_home(game_state)
        
        safe_food = []
        
        #compute safe food positions away from active ghosts
        for food in food_list:
            is_safe = True
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            
            for enemy in enemies:
                if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer == 0:
                    ghost_dist = self.get_maze_distance(food, enemy.get_position())
                    if ghost_dist <= 3:
                        is_safe = False
                        break
            
            if is_safe:
                safe_food.append(food)
        
        #if no safe food and ghost is too close, return home
        if not safe_food and closest_ghost_dist <= 3 and my_state.num_carrying > 0:
            return self.return_home(game_state)
        
        target_food_list = safe_food if safe_food else food_list
        closest_food = min(target_food_list, key=lambda f: self.get_maze_distance(my_pos, f))
        
        #move towards the closest food with ghost avoidance
        return self.move_towards_target(game_state, closest_food, avoid_ghosts=True)


    def choose_action(self, game_state):
        #keep the original choose_action structure and just call the new offensive action function
        return self.get_offensive_action(game_state)


class MitalHybrid(ReflexCaptureAgent):
    """
    The aproach for the final version of our agent Mital is a hybrid agent that switches between offensive and defensive roles.
    What we do is we create a team with 2 mital hybrid agents that share some class variables to keep track of which one is 
    currently offensive and which one is defensive. This way, when one agent dies (respawns at start) or gets stuck in a movement pattern, 
    the other agent takes over the offensive role and the currently dead or stuck agent switches to defensive role. This allows us to 
    maintain continuous pressure on the opponent while ensuring that we have a solid defense in place. It also prevents getting stuck in loops as 
    we will change roles with the other agent.
    """

    #shared class variables to track offensive agent and movement/death history (we use the _ to indicate that these variables are private to the class)
    _offensive_agent_index = None
    _last_death_check = {}
    _move_history = {}
    _position_history = {}

    def __init__(self, index):
        super().__init__(index)
        #sub-agents, created with same index
        self.off_agent = MitalOffensive(index)
        self.def_agent = MitalDefensive(index)

        #set the initial role to None, will be set in register_initial_state
        self.role = None

        #teammate index computed in register_initial_state
        self.teammate_index = None

    def register_initial_state(self, game_state):
        #get the initial state from the parent class
        super().register_initial_state(game_state)

        #register initial state for both sub-agents
        self.off_agent.register_initial_state(game_state)
        self.def_agent.register_initial_state(game_state)

        #discover teammate index, it will depend if we are red or blue team
        team = self.get_team(game_state)
        for idx in team:
            if idx != self.index:
                self.teammate_index = idx
                break

        if MitalHybrid._offensive_agent_index is None:
            #assing initial defensive role based on index
            if self.index < self.teammate_index:
                MitalHybrid._offensive_agent_index = self.teammate_index
                self.role = 'defensive'
            
            #assing the other agent as offensive
            else:
                MitalHybrid._offensive_agent_index = self.index
                self.role = 'offensive'
        
        #if already assigned, set local role accordingly
        else:
            self.role = 'offensive' if MitalHybrid._offensive_agent_index == self.index else 'defensive'

        #we also need to initialize movement and death tracking for both agents
        for idx in (self.index, self.teammate_index):
            if idx not in MitalHybrid._move_history:
                MitalHybrid._move_history[idx] = []
            if idx not in MitalHybrid._position_history:
                MitalHybrid._position_history[idx] = []
            if idx not in MitalHybrid._last_death_check:
                # store initial position
                pos = game_state.get_agent_position(idx)
                MitalHybrid._last_death_check[idx] = pos

    #the hybrid agent is responsible for detecting if the offensive agent is stuck in a movement pattern or if it has died
    def is_stuck_in_pattern(self, position_history, move_history):
        
        #we are only interested in the last 20 moves/positions
        if len(position_history) < 20:
            return False

        recent_positions = position_history[-20:]
        recent_moves = move_history[-20:] if len(move_history) >= 20 else move_history

        #it is important that we dont switch roles if the agent is going in the same direction constantly
        if len(recent_moves) >= 10:
            unique_moves = set(recent_moves[-10:])
            if len(unique_moves) == 1:
                return False
            if len(unique_moves) == 2:
                move_counts = {}
                for m in recent_moves[-10:]:
                    move_counts[m] = move_counts.get(m, 0) + 1
                if max(move_counts.values()) >= 7:
                    return False

        #check position patterns
        unique_positions = set(recent_positions)
        #if only 2 or less unique positions in last 20 moves, definitely stuck
        if len(unique_positions) <= 2:
            return True
        
        # if 3 or 4 unique positions,  check for repetition
        if len(unique_positions) <= 4:
            counts = {}
            for p in recent_positions:
                counts[p] = counts.get(p, 0) + 1
            if any(c >= 5 for c in counts.values()):
                return True
        
        #else, not stuck
        return False

    #funtion to track movement history of the offensive agent
    def update_movement_tracking(self, game_state, action):
        offensive_idx = MitalHybrid._offensive_agent_index
        #only track for the offensive agent
        if self.index != offensive_idx:
            return

        #add current position
        pos = game_state.get_agent_position(self.index)
        ph = MitalHybrid._position_history.setdefault(self.index, [])
        ph.append(pos)
        if len(ph) > 30:
            ph.pop(0)

        #add current action
        mh = MitalHybrid._move_history.setdefault(self.index, [])
        mh.append(action)
        if len(mh) > 30:
            mh.pop(0)

    #function to check if the offensive agent is stuck and switch roles if so
    def check_for_stuck_and_switch_roles(self, game_state):
        offensive_idx = MitalHybrid._offensive_agent_index
        #only for the offensive agent
        if offensive_idx is None:
            return

        #get movement and position history
        pos_hist = MitalHybrid._position_history.get(offensive_idx, [])
        mov_hist = MitalHybrid._move_history.get(offensive_idx, [])

        #call the stuck detection
        if self.is_stuck_in_pattern(pos_hist, mov_hist):
            #swap roles
            new_off = self.teammate_index if offensive_idx == self.index else self.index
            MitalHybrid._offensive_agent_index = new_off

            #we clear histories to avoid changing continuously between roles
            MitalHybrid._move_history[self.index] = []
            MitalHybrid._position_history[self.index] = []
            if self.teammate_index in MitalHybrid._move_history:
                MitalHybrid._move_history[self.teammate_index] = []
            if self.teammate_index in MitalHybrid._position_history:
                MitalHybrid._position_history[self.teammate_index] = []

    #pretty similar to the stuck detection, but now we check if the offensive agent has died (respawned at start)
    def check_for_death_and_switch_roles(self, game_state):
        offensive_idx = MitalHybrid._offensive_agent_index
        #same as before, only for offensive agent
        if offensive_idx is None:
            return

        #obtain offensive agent state
        offensive_state = game_state.get_agent_state(offensive_idx)
        if offensive_state is None:
            return

        #get current and start positions
        offensive_pos = offensive_state.get_position() if hasattr(offensive_state, 'get_position') else None

        offensive_start = None
        start_attr = getattr(offensive_state, 'start', None)
        #check if start attribute exists and has get_position method    
        if start_attr is not None and hasattr(start_attr, 'get_position'):
            offensive_start = start_attr.get_position()

        #check for death (respawn at start)
        if offensive_pos is not None and offensive_start is not None:
            last_known = MitalHybrid._last_death_check.get(offensive_idx)
            
            #if agent was away and now at start -> died & respawned
            if last_known is not None and last_known != offensive_start and offensive_pos == offensive_start:
                new_off = self.teammate_index if offensive_idx == self.index else self.index
                MitalHybrid._offensive_agent_index = new_off
                
                # clear histories for both agents to avoid switching continuously
                MitalHybrid._move_history[self.index] = []
                MitalHybrid._position_history[self.index] = []
                if self.teammate_index in MitalHybrid._move_history:
                    MitalHybrid._move_history[self.teammate_index] = []
                if self.teammate_index in MitalHybrid._position_history:
                    MitalHybrid._position_history[self.teammate_index] = []

        #we update last known position
        for idx in (offensive_idx, self.teammate_index):
            st = game_state.get_agent_state(idx)
            if st is None:
                continue
            pos = st.get_position() if hasattr(st, 'get_position') else None
            MitalHybrid._last_death_check[idx] = pos


    #we need to sync the private role variable with the class variable
    def update_role_from_shared_state(self):
        
        #if this agent is the offensive one
        if MitalHybrid._offensive_agent_index == self.index:
            
            #set role to offensive if not already
            if self.role != 'offensive':
                self.role = 'offensive'
        
        #else, this agent is defensive, so we set it if it was not already
        else:
            if self.role != 'defensive':
                self.role = 'defensive'


    #main choose action function that will delegate to the appropriate sub-agent based on current role
    def choose_action(self, game_state):
        # 1) Check for death/stuck and possibly switch the shared offensive index.
        self.check_for_death_and_switch_roles(game_state)
        self.check_for_stuck_and_switch_roles(game_state)

        # 2) Sync local role
        self.update_role_from_shared_state()

        # 3) Delegate: call the appropriate sub-agent
        if self.role == 'offensive':
            #call offensive agent
            action = self.off_agent.choose_action(game_state)
        else:
            #ensure def_agent camping positions exist
            if not getattr(self.def_agent, 'camping_pos', None):
                #call camping to recompute teh camping positions
                try:
                    self.def_agent.camping(game_state)
                except Exception:
                    pass
            #call defensive agent
            action = self.def_agent.choose_action(game_state)

        # 4) Track movement history
        self.update_movement_tracking(game_state, action)

        return action