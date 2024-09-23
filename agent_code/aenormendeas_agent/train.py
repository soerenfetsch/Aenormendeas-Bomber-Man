from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, Q_TABLE_FILE, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
Scores = []
Wins = []
Other_Scores = []

SCORE_FILE = "classic_versus_collector_small_scores2.pkl"
WINS_FILE = "classic_versus_collector_small_wins2.pkl"
OTHERS_SCORE_FILE = "classic_collector_small_scores2.pkl"

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
# Custom events
MOVE_CLOSER_TO_COIN = "MOVE_CLOSER_TO_COIN"
MOVE_AWAY_FROM_COIN = "MOVE_AWAY_FROM_COIN"
MOVE_CLOSER_TO_CRATE = "MOVE_CLOSER_TO_CRATE"
MOVE_AWAY_FROM_CRATE = "MOVE_AWAY_FROM_CRATE"
DROPPED_BOMB_AT_CRATE = "DROPPED_BOMB_AT_CRATE"
DROPPED_BOMB_AT_OPPONENT = "DROPPED_BOMB_AT_OPPONENT"
NOTHING_HAPPENED = "NOTHING_HAPPENED"
INEFFECTIVE_BOMB = 'INEFFECTIVE_BOMB'
MOVE_CLOSER_TO_EXPLOSION = 'MOVE_CLOSER_TO_EXPLOSION'
MOVED = 'MOVED'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # replay buffer to store experiences
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # set up training parameters for exploration and learning
    # TODO: Maybe do epsilon decay starting with lots of exploration?
    self.epsilon = 0.6
    self.gamma = 0.5
    self.alpha = 0.2
    self.epsilon_decay = 0.999
    self.min_epsilon = 0.15

    self.game_rewards = {
        e.COIN_COLLECTED: 1200.0,
        e.KILLED_SELF: -3500.0,
        e.GOT_KILLED: -3500.0,
        PLACEHOLDER_EVENT: 0.0,
        e.INVALID_ACTION: -1000.0,
        MOVE_CLOSER_TO_COIN: 90.0,
        MOVE_AWAY_FROM_COIN: -120.0,
        MOVE_CLOSER_TO_CRATE: 70.0,
        MOVE_AWAY_FROM_CRATE: -80.0,
        e.WAITED: -50.0,
        e.CRATE_DESTROYED: 120.0,
        DROPPED_BOMB_AT_CRATE: 400.0,
        DROPPED_BOMB_AT_OPPONENT: 300.0,
        NOTHING_HAPPENED: -2.0,
        INEFFECTIVE_BOMB: -150,
        MOVE_CLOSER_TO_EXPLOSION: -3000.0,
        MOVED: -3.0,
        e.KILLED_OPPONENT: 2000.0
    }

def update_q_values(self):
    """ 
    Update Q-Values based on the Bellman equation by sampling batches
    from the transitions buffer.
    
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    while self.transitions:
        transition = self.transitions.popleft()
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state

        # Get Q-Value from table else get 0.0
        q_value = self.q_table.get(tuple(state), {}).get(action, 0.0)
        # Update the Q-Value based on Bellman equation update rule
        if next_state is None:
            new_reward = reward
        else:
            next_q_values: dict = self.q_table.get(tuple(next_state),
                                             {a: 0.0 for a in ACTIONS})
            max_next_q_value = max(next_q_values.values())
            new_reward = reward + self.gamma * max_next_q_value
        new_q_value = q_value + self.alpha * (new_reward - q_value)

        # Update the Q-table
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = {}
        self.q_table[tuple(state)][action] = new_q_value
        # print('UPDATING Q-VALUES OF ACTION', action)
        # print('OLD Q-VALUE =', q_value)
        # print('NEW Q-VALUE =', new_q_value)

    # Save the Q-Table as pkl file
    with open(Q_TABLE_FILE, "wb") as file:
        pickle.dump(self.q_table, file)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    old_features, old_objective_distances = state_to_features(self, old_game_state)
    new_features, new_objective_distances = state_to_features(self, new_game_state)

    old_coin_distances = old_objective_distances[0]
    new_coin_distances = new_objective_distances[0]

    old_crate_distances = old_objective_distances[1]
    new_crate_distances = new_objective_distances[1]

    global oldold_position
    # Custom events to hand out reward
    # event for moving closer to closest coin / farther away
    if e.COIN_COLLECTED not in events:
        if new_coin_distances < old_coin_distances:
            events.append(MOVE_CLOSER_TO_COIN)
        elif new_coin_distances > old_coin_distances:
            events.append(MOVE_AWAY_FROM_COIN)
    if e.COIN_COLLECTED not in events and e.BOMB_DROPPED not in events:
        if new_crate_distances < old_crate_distances:
            events.append(MOVE_CLOSER_TO_CRATE)
        elif new_crate_distances > old_crate_distances:
            events.append(MOVE_AWAY_FROM_CRATE)
    if e.BOMB_DROPPED in events and old_features[13] > 0:
        events.append(DROPPED_BOMB_AT_CRATE)
    if e.BOMB_DROPPED in events and old_features[14] == 1:
        events.append(DROPPED_BOMB_AT_OPPONENT)
    if e.COIN_COLLECTED not in events and e.CRATE_DESTROYED and e.KILLED_OPPONENT not in events:
        events.append(NOTHING_HAPPENED)
    if e.BOMB_EXPLODED in events and (e.CRATE_DESTROYED not in events and 
                                      e.KILLED_OPPONENT not in events):
        events.append(INEFFECTIVE_BOMB)
    if e.MOVED_LEFT in events and old_features[7] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    if e.MOVED_RIGHT in events and old_features[8] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    if e.MOVED_UP in events and old_features[9] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    if e.MOVED_DOWN in events and old_features[10] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    if e.WAITED in events and old_features[11] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    if e.BOMB_DROPPED in events and old_features[12] == 1:
        events.append(MOVE_CLOSER_TO_EXPLOSION)
    # Any movement
    if (e.MOVED_UP in events or e.MOVED_DOWN in events
        or e.MOVED_LEFT in events or e.MOVED_RIGHT in events):
        events.append(MOVED)
    oldold_position = old_game_state['self'][3]
    # print(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_features, self_action, new_features, reward_from_events(self, events)))

    # Gradually decrease epsilon
    self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

def save_game_score(last_game_state):
    """
    Save the score of the agent at the end of a game to a file
    
    :param last_game_state (dict): last game state occured
    """
    score = last_game_state['self'][1]
    Scores.append(score)

    with open(SCORE_FILE, 'wb') as file:
        pickle.dump(Scores, file)

def save_game_winner(last_game_state):
    """
    Save if the agent won the game to a file.
    1 if the agent won/tied, 0 if not.
    
    :param last_game_state (dict): last game state occured
    """
    won = all([last_game_state['self'][1] > other[1]
           for other in last_game_state['others']])
    Wins.append(won)
    with open(WINS_FILE, 'wb') as file:
        pickle.dump(Scores, file)

def save_others_game_score(last_game_state):
    """
    Save the score of the opponent agents at the end of a game to a file
    
    :param last_game_state (dict): last game state occured
    """
    scores = tuple([other[1] for other in last_game_state['others']])
    Other_Scores.append(scores)

    with open(OTHERS_SCORE_FILE, 'wb') as file:
        pickle.dump(Other_Scores, file)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # print(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Keep track of the number of coins collected
    save_game_score(last_game_state)
    # Keep track of the winner
    save_game_winner(last_game_state)
    # Save others' game scores
    save_others_game_score(last_game_state)

    # Update the Q Values
    last_features, _, = state_to_features(self, last_game_state)
    self.transitions.append(Transition(
        last_features, last_action, None, reward_from_events(self, events)
    ))
    update_q_values(self)

def reward_from_events(self, events: List[str]) -> int:
    """
    Reward or punish the agent depending on the occurred events. The game_rewards
    is created in the setup_training method and can be adjusted there. Returns
    the sum of all rewards.

    :param self: The same object that is passed to all of your callbacks.
    :param events (List[str]): list of events occurred in the last step
    :return sum (float): sum of all rewards for all events occurred
    """
    reward_sum = 0
    for event in events:
        if event in self.game_rewards:
            reward_sum += self.game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # print(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
