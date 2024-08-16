import os
import sys
import pickle
import random
import numpy as np
import heapq

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(SRC_DIR)
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

N_CLOSEST_COINS = 1  # number of closest coins to consider for features
N_CLOSEST_CRATES = 1  # number of closest crates to consider for features

Q_TABLE_FILE = "coins_crates_bombs_qtable.pkl"
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
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    self.logger.debug("Exploitation: Selecting best action using Q-Value.")
    features, _, __ = state_to_features(self, game_state)
    q_values: dict = self.q_table.get(tuple(features),
                                      {a: 0.0 for a in ACTIONS})
    # print(f"Given features: {features}, Q-values: {q_values}")
    max_q_value = max(q_values.values())
    best_actions = [a for a, q in q_values.items() if q == max_q_value]
    return np.random.choice(best_actions)


def adjacent_tile_features(self, game_state, agent_position):
    """ 
    Convert the agent's position and the game state to features for the 4 adjacent
    tiles UP, DOWN, LEFT, RIGHT. We encode each tile in binary:
    0: outside bound, 1: stone wall, 2: crate,
    3: free tile

    :param: game_state (dict): The dictionary that describes everything on the board.
    :agent_position (np.array([x, y])): current coordinates of the agent
    :return: np.array(list)[4]: concatenated tile features UP, DOWN, LEFT, RIGHT
    """
    self.logger.debug("Converting adjacent fields to features")
    y, x = agent_position
    height, width = game_state['field'].shape
    tile_features = []
    # UP DOWN LEFT RIGHT
    directions = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    for dy, dx in directions:
        # Outside bound
        if not (0 <= dx < width and 0 <= dy < height):
            tile_features.append(0)
        # Stone wall
        elif game_state['field'][dy, dx] == -1:
            tile_features.append(1)
        # Crate
        elif game_state['field'][dy, dx] == 1:
            tile_features.append(2)
        # Free tile
        elif game_state['field'][dy, dx] == 0:
            tile_features.append(3)

    assert len(tile_features) == 4
    return np.array(tile_features)

# TODO: add objectives crates and bombs


def find_closest_objectives(self, game_state, agent_position):
    """
    Use Dijkstra's algorithm to construct shortest path search
    and find the closest coins/crates/bombs up the the max number
    specified in the setup. We construct a parent tree to store
    the movement UP, DOWN, LEFT, RIGHT adjacent to the agent as
    well as the respective distance as a feature for each objective.
    WAIT: -1, UP: 0, DOWN: 1, LEFT: 2, RIGHT: 3. The distance is only
    saved up to the MAX_SAVE_DIST to lower the number of states.
    :params: game_state (dict): The dictionary that describes everything on the board.
    :agent_position (np.array([x, y])): current coordinates of the agent
    :return: np.array(list): concatenated tile features for all objectives
             np.array(list): concatenated list of actual distances to objectives
    """
    self.logger.debug("Finding closest coin objectives.")
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
    max_crates = min(N_CLOSEST_CRATES, np.count_nonzero(game_state['field'] == 1))
    while not len(q) == 0 and (coins_picked < max_coins or crates_picked < max_crates):
        curr_dist, (dy, dx) = heapq.heappop(q)
        if curr_dist > distances[dx, dy]:
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
        if crates_picked < max_crates and game_state['field'][dy, dx] == 1:
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
        directions = [(dy-1, dx), (dy+1, dx), (dy, dx-1), (dy, dx+1)]

        for ddy, ddx in directions:
            if not (0 <= ddx < width and 0 <= ddy < height):
                continue
            if game_state['field'][ddy, ddx] not in (0, 1):
                continue
            if curr_dist + 1 < distances[ddy, ddx]:
                distances[ddy, ddx] = curr_dist + 1
                parents[ddy, ddx] = (dy, dx)
                heapq.heappush(q, (curr_dist+1, (ddy, ddx)))
    return (np.concatenate([out_features_coins, out_features_crates]),
            np.concatenate([coin_distances, crate_distances]))


def adjacent_explosions(self, game_state, agent_position):
    '''
    if 2 explosions are planned for 1 adjacent block, 
    will only consider the explosion that happens sooner.
    value of feature says how long until explosion over, meaning 
    if bomb not exploded yet
    value is bomb timer + explosion timer

    walls stop explosion
    '''
    self.logger.debug("Converting adjacent fields to features")
    y, x = agent_position
    height, width = game_state['explosion_map'].shape
    explosion_features = []
    # UP DOWN LEFT RIGHT HERE
    look = [(y-1, x), (y+1, x), (y, x-1), (y, x+1),(y,x)]
    future_explosions = []
    #problem: explosion doesnt go through walls
    for b in game_state['bombs']:
        y,x,t=b[0][0],b[0][1],b[1]+s.EXPLOSION_TIMER
        future_explosions.append(b)
        for dy in range(1,s.BOMB_POWER):
            if game_state['field'][y+dy][x]==-1:
                break
            else:
                future_explosions.append([(y+dy,x),t])
        for dy in range(1,s.BOMB_POWER):
            if game_state['field'][y-dy][x]==-1:
                break
            else:
                future_explosions.append([(y-dy,x),t])
        for dx in range(1,s.BOMB_POWER):
            if game_state['field'][y][x+dx]==-1:
                break
            else:
                future_explosions.append([(y,x+dx),t])
        for dy in range(1,s.BOMB_POWER):
            if game_state['field'][y][x-dx]==-1:
                break
            else:
                future_explosions.append([(y,x-dx),t])

    for dy, dx in look:
        if game_state['explosion_map'][dy][dx]!=0:
            explosion_features.append(game_state['explosion_map'][dy][dx])
        else:
            #fehler
            explosions_at_dxdy = []
            for e in future_explosions:
                if e[0] == (dy,dx):
                    explosions_at_dxdy.append(e[1])
            if len(explosions_at_dxdy)!=0:
                explosion_features.append(min(explosions_at_dxdy))
            else:
                explosion_features.append(0)

    assert len(explosion_features) == 5
    return np.array(explosion_features)
     

def escape(self, game_state, agent_position):
    
    self.logger.debug("Finding closest coin objectives.")
    y, x = agent_position
    # UP DOWN LEFT RIGHT
    if game_state['explosion_map'][y,x]==0:
        return np.array([0]),0
    neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
    height, width = game_state['field'].shape
    distances = np.ones(game_state['field'].shape) * float('inf')
    distances[y, x] = 0
    parents = np.zeros(game_state['field'].shape,
                       dtype=np.dtype([('y', int), ('x', int)]))
    parents[y, x] = (y, x)
    q = [(0, (y, x))]
    escapes_picked = 0
    out_features_escapes = [-1 for _ in range(1)]
    escape_distances = [float('inf') for _ in range(1)]
    max_escapes = min(1, np.sum((game_state['field']==0)==(game_state['explosion_map']==0),axis=(0,1)))
    while not len(q) == 0 and (escapes_picked < max_escapes):
        curr_dist, (dy, dx) = heapq.heappop(q)
        if curr_dist > distances[dx, dy]:
            continue
        # If coin is found
        if escapes_picked < max_escapes and game_state['explosion_map'][dy,dx]==0:
            if (dy, dx) == (y, x):
                out_features_escapes[escapes_picked] = -1
                escape_distances[escapes_picked] = curr_dist
                escapes_picked += 1
            else:
                ddy, ddx = dy, dx
                while (ddy, ddx) not in neighbors:
                    ddy, ddx = parents[ddy, ddx]
                index = neighbors.index((ddy, ddx))
                out_features_escapes[escapes_picked] = index
                escape_distances[escapes_picked] = curr_dist
                escapes_picked += 1
        
        directions = [(dy-1, dx), (dy+1, dx), (dy, dx-1), (dy, dx+1)]

        for ddy, ddx in directions:
            if not (0 <= ddx < width and 0 <= ddy < height):
                continue
            if game_state['field'][ddy, ddx] not in (0, 1):
                continue
            if curr_dist + 1 < distances[ddy, ddx]:
                distances[ddy, ddx] = curr_dist + 1
                parents[ddy, ddx] = (dy, dx)
                heapq.heappush(q, (curr_dist+1, (ddy, ddx)))
    return np.array(out_features_escapes), escape_distances

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
    escape_dir,escape_dist = escape(self, game_state, agent_position)
    return np.concatenate([adjacent_tile_features(self, game_state, agent_position),
                           adjacent_explosions(self, game_state, agent_position),
                           objective_features,
                           escape_dir]), objective_distances, escape_dist


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
                                   [0, 1, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0, 0]])
    }
    features, objective_distances, escape_distance = state_to_features(object(), game_state)
    print(features)
    print(objective_distances)
    print(features[:4])
    assert (len(features) == len(np.array([1, 1, 3, 3, 0, 0, 1, 3, 0, 3, 3, 3])))
    assert (features == np.array([1, 1, 3, 3, 0, 0, 1, 3, 0, 3, 3, 3])).all()