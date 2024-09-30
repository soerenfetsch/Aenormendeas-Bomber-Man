import os
import sys
import pickle
import copy
import numpy as np
import heapq

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(SRC_DIR)
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

N_CLOSEST_COINS = 1  # number of closest coins to consider for features
N_CLOSEST_CRATES = 1  # number of closest crates to consider for features

Q_TABLE_FILE = "minimal_aenormendeas.pkl"
TRAIN_NEW = False


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not TRAIN_NEW:
        qtable_file = Q_TABLE_FILE
    else:
        qtable_file = None
    if TRAIN_NEW and self.train or not os.path.isfile(qtable_file):
        self.logger.info("Setting up Q-Table from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Loading Q-Table from saved state.")
        with open(qtable_file, "rb") as file:
            self.q_table = pickle.load(file)
        self.logger.debug(
            f"Loaded saved Q-Table with {len(self.q_table)} entries.")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    We use an epsilon parameter during training for exploration. Otherwise convert
    the game state to a feature vector and select the best actions from the q_table
    given the feature_vector as key.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Exploration vs exploitation
    if game_state is None or (self.train and np.random.rand() < self.epsilon):
        self.logger.debug(
            "Epsilon Exploration: Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    self.logger.debug("Exploitation: Selecting best action using Q-Value.")
    features, _, = state_to_features(self, game_state)
    q_values: dict = self.q_table.get(tuple(features),
                                      {a: 0.0 for a in ACTIONS})
    # print(f"Given features: {features}, Q-values: {dict(sorted(
    #     q_values.items(), key=lambda x: x[1], reverse=True))}")
    max_q_value = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q_value]
    return np.random.choice(best_actions)

def get_adv_bomb_field(self, game_state):
    """ 
    Get a matrix of the explosion state including future explosions
    of placed bombs taking the length of the explosion at every position
    where there is an explosion or adding the length of an entire explosion
    to the minimum bomb timer of a field where a bomb will explode in the
    future.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: map (np.array): map of field size with every entry as explained
    """
    original_field = game_state['explosion_map']

    future_explosions = []
    for b in game_state['bombs']:
        y, x, t = b[0][0], b[0][1], b[1]+s.EXPLOSION_TIMER
        future_explosions.append([(y, x), t])
        for dy in range(1, s.BOMB_POWER+1):
            if game_state['field'][y+dy][x] == -1:
                break
            else:
                future_explosions.append([(y+dy, x), t])
        for dy in range(1, s.BOMB_POWER+1):
            if game_state['field'][y-dy][x] == -1:
                break
            else:
                future_explosions.append([(y-dy, x), t])
        for dx in range(1, s.BOMB_POWER+1):
            if game_state['field'][y][x+dx] == -1:
                break
            else:
                future_explosions.append([(y, x+dx), t])
        for dx in range(1, s.BOMB_POWER+1):
            if game_state['field'][y][x-dx] == -1:
                break
            else:
                future_explosions.append([(y, x-dx), t])
    for coord, t in future_explosions:
        # Use <= 0 for altered t_maps where 1 is iteratively subtracted, which
        # results in possible negative numbers (e.g. is_bombing_deadly function)
        if original_field[coord] <= 0 or t < original_field[coord]:
            original_field[coord] = t

    return original_field

def adjacent_walkable_tiles(self, game_state, agent_position):
    """
    Convert the agent's position and the game state to features for the 4 adjacent
    tiles UP, DOWN, LEFT, RIGHT (matrix style not gui, in gui it is LEFT, RIGHT, UP, DOWN).
    We encode each tile:
    0: not walkable (stone wall or crate), 1: walkable (free tile)

    :param: game_state (dict): The dictionary that describes everything on the board.
    :param: agent_position (np.array([x, y])): current coordinates of the agent
    :return: np.array(list)[4]: concatenated tile features UP, DOWN, LEFT, RIGHT
    """
    y, x = agent_position
    height, width = game_state['field'].shape
    tile_features = []
    # UP DOWN LEFT RIGHT
    directions = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    for dy, dx in directions:
        # Free tile and no bomb or other player on tile
        if (game_state['field'][dy, dx] == 0 and 
            (dy, dx) not in [o[3] for o in game_state['others']] and
            (dy, dx) not in [b[0] for b in game_state['bombs']]):
            tile_features.append(1)
        else:
            tile_features.append(0)

    assert len(tile_features) == 4
    return np.array(tile_features)

def find_closest_objectives(self, game_state, agent_position):
    """
    Use Dijkstra's algorithm to construct shortest path search
    and find the closest coins/crates/bombs up the the max number
    specified in the setup. We construct a parent tree to store
    the movement UP, DOWN, LEFT, RIGHT adjacent to the agent as
    well as the respective distance as a feature for each objective.
    WAIT: -1, UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3. The distance is only
    saved up to the MAX_SAVE_DIST to lower the number of states.
    New: path to objectives now only take a path along free tiles, if
    there is a crate, the path is blocked.

    :param game_state (dict): The dictionary that describes everything on the board.
    :param agent_position (np.array): current coordinates of the agent
    :return np.array(list): concatenated tile features for all objectives
    :return np.array(list): concatenated list of actual distances to objectives
    """
    y, x = agent_position
    # UP DOWN LEFT RIGHT
    neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    height, width = game_state['field'].shape
    distances = np.ones(game_state['field'].shape) * float('inf')
    distances[y, x] = 0
    parents = np.zeros(game_state['field'].shape,
                       dtype=np.dtype([('y', int), ('x', int)]))
    parents[y, x] = (y, x)
    q = [(0, (y, x))]
    coins_picked = 0
    crates_picked = 0
    out_features_coins = [-1 for _ in range(N_CLOSEST_COINS)]
    coin_distances = [float('inf') for _ in range(N_CLOSEST_COINS)]
    out_features_crates = [-1 for _ in range(N_CLOSEST_CRATES)]
    crate_distances = [float('inf') for _ in range(N_CLOSEST_CRATES)]
    max_coins = min(N_CLOSEST_COINS, len(game_state['coins']))
    max_crates = min(N_CLOSEST_CRATES, np.count_nonzero(
        game_state['field'] == 1))
    adv_bomb_map = get_adv_bomb_field(self, game_state)
    
    enemies_picked = 0
    out_features_enemies = [-1 for _ in range(N_CLOSEST_CRATES)]
    enemy_distances = [float('inf') for _ in range(N_CLOSEST_CRATES)]
    max_enemies = min(N_CLOSEST_CRATES, len(game_state['others']))
    
    while not len(q) == 0 and (coins_picked < max_coins or 
                               crates_picked < max_crates or
                               enemies_picked < max_enemies):
        curr_dist, (dy, dx) = heapq.heappop(q)
        if curr_dist > distances[dy, dx]:
            continue
        # If coin is found
        if coins_picked < max_coins and (dy, dx) in game_state['coins']:
            if (dy, dx) == (y, x):
                out_features_coins[coins_picked] = -1
                coin_distances[coins_picked] = curr_dist
                coins_picked += 1
            else:
                ddy, ddx = dy, dx
                while (ddy, ddx) not in neighbors:
                    ddy, ddx = parents[ddy, ddx]
                index = neighbors.index((ddy, ddx))
                out_features_coins[coins_picked] = index
                coin_distances[coins_picked] = curr_dist
                coins_picked += 1
        # If crate is found
        if (crates_picked < max_crates and 
            game_state['field'][dy, dx] == 1 and 
            adv_bomb_map[dy, dx] <= 0):
            if (dy, dx) == (y, x):
                out_features_crates[crates_picked] = -1
                crate_distances[crates_picked] = curr_dist
                crates_picked += 1
            else:
                ddy, ddx = dy, dx
                while (ddy, ddx) not in neighbors:
                    ddy, ddx = parents[ddy, ddx]
                index = neighbors.index((ddy, ddx))
                out_features_crates[crates_picked] = index
                crate_distances[crates_picked] = curr_dist
                crates_picked += 1
        # If enemy is found
        if enemies_picked < max_enemies and (dy, dx) in [e[3] for e in game_state['others']]:
            if (dy, dx) == (y, x):
                out_features_enemies[enemies_picked] = -1
                enemy_distances[enemies_picked] = curr_dist
                enemies_picked += 1
            else:
                ddy, ddx = dy, dx
                while (ddy, ddx) not in neighbors:
                    ddy, ddx = parents[ddy, ddx]
                index = neighbors.index((ddy, ddx))
                out_features_enemies[enemies_picked] = index
                enemy_distances[enemies_picked] = curr_dist
                enemies_picked += 1
        directions = [(dy-1, dx), (dy+1, dx), (dy, dx-1), (dy, dx+1)]
        # If tile was crate do not go along this path further
        if game_state['field'][dy, dx] == 1:
            continue
        for ddy, ddx in directions:
            if not (0 <= ddx < width and 0 <= ddy < height):
                continue
            if game_state['field'][ddy, ddx] not in (0, 1):
                continue
            if curr_dist + 1 < distances[ddy, ddx]:
                distances[ddy, ddx] = curr_dist + 1
                parents[ddy, ddx] = (dy, dx)
                heapq.heappush(q, (curr_dist+1, (ddy, ddx)))
    return (np.concatenate([out_features_coins, out_features_crates, out_features_enemies]),
            np.concatenate([coin_distances, crate_distances, enemy_distances]))

def is_crate_in_range(self, game_state, agent_position):
    """
    Feature to note if a crate is in bombing range.
    
    :param game_state (dict): The dictionary that describes everything on the board.
    :param agent_position (np.array): current coordinates of the agent
    :return np.array(int): [1] if crate in range, [0] if not
    """
    y, x = agent_position
    height, width = game_state['explosion_map'].shape
    crate_count = game_state['field'][y][x]
    for dy in range(1, s.BOMB_POWER+1):
        if y+dy >= height or game_state['field'][y+dy][x] == -1:
            break
        if game_state['field'][y+dy][x] == 1:
            return np.array([1])
    for dy in range(1, s.BOMB_POWER+1):
        if y-dy < 0 or game_state['field'][y-dy][x] == -1:
            break
        if game_state['field'][y-dy][x] == 1:
            return np.array([1])
    for dx in range(1, s.BOMB_POWER+1):
        if x+dx >= width or game_state['field'][y][x+dx] == -1:
            break
        if game_state['field'][y][x+dx] == 1:
            return np.array([1])
    for dx in range(1, s.BOMB_POWER+1):
        if x-dx < 0 or game_state['field'][y][x-dx] == -1:
            break
        if game_state['field'][y][x-dx] == 1:
            return np.array([1])
    return np.array([0])

def get_neighbors(pos, game_state, advanced_bomb_field, dist, can_pass_bomb=False):
    # removes 1 tick from explosion time since you need 1 tick to reach the neighbor
    y, x = pos
    # All 5 'neighbors' with UP DOWN LEFT RIGHT HERE
    neighbors = [[(y-1, x), advanced_bomb_field[y-1, x] - 1, dist+1, False],
                 [(y+1, x), advanced_bomb_field[y+1, x] - 1, dist+1, False],
                 [(y, x-1), advanced_bomb_field[y, x-1] - 1, dist+1, False],
                 [(y, x+1), advanced_bomb_field[y, x+1] - 1, dist+1, False],
                 [(y, x), advanced_bomb_field[y, x] - 1, dist+1, can_pass_bomb]]
    return neighbors

def deadly_adjacent_fields(self, game_state, agent_position, adv_bomb_field):
    """
    Find out whether adjacent fields of the agent are deadly, meaning that
    going there will yield certain death as that future explosions
    cannot be circumvented.

    :param game_state:  A dictionary describing the current game board
    :param agent_position (np.array([x, y])): current coordinates of the agent
    :param adv_bomb_field (np.array): bomb field with bomb and time info of explosions
    :param can_pass_bomb (bool): When dropping a bomb the agent can remain on the bomb
           while stepping to the side means it cannot be passed anymore
    :return deadly (np.array): length 5, UP DOWN LEFT RIGHT HERE, 1 for deadly, 0 not
    """
    y, x = agent_position
    # UP DOWN LEFT RIGHT HERE
    neighbors = get_neighbors((y, x), game_state, adv_bomb_field, 0, True)

    deadly = np.array([1, 1, 1, 1, 1])
    if adv_bomb_field[y, x] == 0:
        deadly[4] = 0

    height, width = game_state['field'].shape
    for i in range(5):
        # crates could be destroyed when reached
        dynamic_field = np.copy(game_state['field'])
        cur = [neighbors[i]]
        t_map = np.copy(adv_bomb_field)
        t_map -= 1
        current_dist = 1
        while len(cur) != 0:
            # pop out cur with shortest distance
            shortest_dist = min(cur, key=lambda x: x[2])
            pos, t, dist, can_pass_bomb = shortest_dist
            dy, dx = pos
            cur.remove([pos, t, dist, can_pass_bomb])
            # update t_map if necessary
            if dist > current_dist:
                assert current_dist == dist-1
                current_dist += 1
                t_map -= 1
            # check if crate will still exist
            if dynamic_field[dy, dx] == 1:
                if t_map[dy, dx] < s.EXPLOSION_TIMER and adv_bomb_field[dy, dx] > 0:
                    dynamic_field[dy, dx] = 0

            if dynamic_field[dy, dx] == 0:
                # bomb cannot be passed through if you have stepped away from it once
                passable_bomb = can_pass_bomb or (dy, dx) not in [b[0] for b in game_state['bombs']]
                # t < 0 means spot is safe
                if t < 0:
                    deadly[i] = 0
                    break
                elif t >= s.EXPLOSION_TIMER and passable_bomb:
                    # t >= explosion timer means spot is not lethal YET and can be passed through
                    # if there is not a bomb on the spot
                    cur.extend(get_neighbors(pos, game_state, t_map, dist, can_pass_bomb))
                # else spot currently exploding and/or cannot be traversed
    return deadly

def is_bombing_deadly(self, game_state, agent_position, adv_bomb_field):
    """ 
    Given the current game_state calculates whether using the action BOMB
    would cause certain death for the player.

    :param game_state:  A dictionary describing the current game board
    :param agent_position (np.array([x, y])): current coordinates of the agent
    :param adv_bomb_field (np.array): bomb field with bomb and time info of explosions
    :return deadly (np.array): [1] if deadly, [0] if not
    """
    new_game_state = copy.deepcopy(game_state)
    bomb = [tuple(agent_position), s.BOMB_TIMER+1]
    new_game_state['bombs'].append(bomb)
    new_adv_bomb_field = get_adv_bomb_field(self, new_game_state)
    
    deadly_features = deadly_adjacent_fields(
        self, new_game_state, agent_position, new_adv_bomb_field)
    return np.array([deadly_features[4]])

def is_enemy_in_range(self, game_state, agent_position):
    """
    Checks if any enemy is in bombing range. 

    :param game_state:  A dictionary describing the current game board
    :param agent_position (np.array([x, y])): current coordinates of the agent
    :return 1 if enemy is in range, 0 if not
    """
    y, x = agent_position
    if len(game_state['others'])==0:
        return np.array([0])
    enemies = [e[3] for e in game_state['others']]
    height, width = game_state['field'].shape
    for dy in range(1, s.BOMB_POWER+1):
        if y+dy >= height or game_state['field'][y+dy][x] == -1:
            break
        if (y+dy,x) in enemies :
            return np.array([1])
    for dy in range(1, s.BOMB_POWER+1):
        if y-dy < 0 or game_state['field'][y-dy][x] == -1:
            break
        if (y-dy,x) in enemies :
            return np.array([1])
    for dx in range(1, s.BOMB_POWER+1):
        if x+dx >= width or game_state['field'][y][x+dx] == -1:
            break
        if (y, x+dx) in enemies :
            return np.array([1])
    for dx in range(1, s.BOMB_POWER+1):
        if x-dx < 0 or game_state['field'][y][x-dx] == -1:
            break
        if (y, x-dx) in enemies :
            return np.array([1])
    return np.array([0])

def can_place_bomb(self, game_state):
    """
    Checks if agent can bomb

    :param game_state:  A dictionary describing the current game board
    :return [1] if agent can bomb, [0] if not
    """
    if game_state['self'][2]:
        return [1]
    else:
        return [0]

def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: features (np.array),
             objective_distances (np.array)
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    agent_position = game_state['self'][3]
    objective_features, objective_distances = find_closest_objectives(
        self, game_state, agent_position)
    deadly_adjacent_features = deadly_adjacent_fields(
        self, game_state, agent_position, get_adv_bomb_field(self, game_state))
    can_bomb = is_bombing_deadly(
        self, game_state, agent_position, get_adv_bomb_field(self, game_state))
    return (np.concatenate([adjacent_walkable_tiles(self, game_state, agent_position),
                           objective_features,
                           deadly_adjacent_features,
                           can_bomb,
                           is_crate_in_range(self, game_state, agent_position),
                           is_enemy_in_range(self, game_state, agent_position),
                           can_place_bomb(self, game_state)]),
            objective_distances)


if __name__ == "__main__":
    class object:
        class logg:
            def debug(self, str):
                print(str)
        logger = logg()
    # Test the feature vector
    game_state = {
        'round': 0,
        'step': 0,
        'field': np.array([[-1, -1, -1, -1, -1, -1],
                          [-1, 0, 0, 0, -1, -1],
                          [-1, 1, 0, 0, 0, -1],
                          [-1, -1, -1, 0, -1, -1],
                          [-1, 0, 0, 0, 0, -1],
                          [-1, -1, -1, -1, -1, -1]]),
        'coins': [(1, 1), (2, 3), (2, 4), (4, 4)],
        'self': ("Name", 0, True, (4, 2)),
        'bombs': [[(1, 1), 3], [(2, 3), 1]],
        'explosion_map': np.array([[0, 0, 0, 0, 0, 0],
                                   [0, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]]),
        'others': []
    }
    features, objective_distances = state_to_features(
        object(), game_state)
    print(features)
    print(objective_distances)
    assert (len(features) == len(
        np.array([0, 0, 1, 1, 3, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 1])))
    assert (features == np.array(
        [0, 0, 1, 1, 3, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 1])).all()

    # Test the advanced bomb map
    adv_bomb_map = get_adv_bomb_field(object(), game_state)
    print(adv_bomb_map)
    assert adv_bomb_map.shape == game_state['field'].shape
    assert (adv_bomb_map == np.array([[0, 0, 0, 0, 0, 0],
                                      [0, 1, 1, 3, 0, 0],
                                      [0, 3, 1, 1, 3, 0],
                                      [0, 0, 0, 3, 0, 0],
                                      [0, 1, 0, 3, 0, 0],
                                      [0, 0, 0, 0, 0, 0]])).all()

    # Test the deadly adjacent field features and is_bombing_deadly
    game_state_0 = {
        'round': 0,
        'step': 0,
        'field': np.array([[-1, -1, -1, -1, -1, -1],
                           [-1, 0, 0, 0, 0, -1],
                           [-1, -1, -1, 0, -1, -1],
                           [-1, 0, 0, 0, 0, -1],
                           [-1, 0, 0, 1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]]),
        'coins': [],
        'self': ("Name", 0, True, (3, 3)),
        'bombs': [[(3, 2), 4], [(4, 1), 1]],
        'explosion_map': np.array([[0, 0, 0, 0, 0, 0],
                                   [0, 0, 2, 2, 2, 0],
                                   [0, 0, 0, 2, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]]),
        'others': [('enemyA', 100, 0, (1, 1)),
                   ('enemyB', 50, 1, (3, 1))]
    }
    deadly = deadly_adjacent_fields(object(), game_state_0, (3, 3),
                                    get_adv_bomb_field(object(), game_state_0))
    print(deadly)
    assert (deadly == np.array([1, 1, 0, 0, 0])).all()
    deadly_bombing = is_bombing_deadly(object(), game_state_0, (3, 3),
                                    get_adv_bomb_field(object(), game_state_0))
    print(deadly_bombing)
    assert (deadly_bombing == np.array([0])).all()
    enemy_in_range = is_enemy_in_range(object(), game_state_0, (3, 3))
    assert enemy_in_range == np.array([1])
    feature_enemy, enemy_distance = [e[2] for e in find_closest_objectives(object(), game_state_0, (3, 3))]
    print(feature_enemy, enemy_distance)
    assert feature_enemy == 2
    assert enemy_distance == 2