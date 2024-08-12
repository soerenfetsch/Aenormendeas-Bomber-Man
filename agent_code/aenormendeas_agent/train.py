from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, N_CLOSEST_COINS, Q_TABLE_FILE, ACTIONS

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
# Custom events
MOVE_CLOSER_TO_COIN = "MOVE_CLOSER_TO_COIN"
MOVE_AWAY_FROM_COIN = "MOVE_AWAY_FROM_COIN"


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
    self.epsilon = 0.2
    self.gamma = 0.9
    self.alpha = 0.1

    # TODO: Add more rewards for upcoming tasks
    self.game_rewards = {
        e.COIN_COLLECTED: 150.0,
        e.KILLED_SELF: -50.0,
        e.GOT_KILLED: -200.0,
        PLACEHOLDER_EVENT: 0.0,
        e.INVALID_ACTION: -200.0,
        MOVE_CLOSER_TO_COIN: 30.0,
        MOVE_AWAY_FROM_COIN: -10.0,
        e.WAITED: 0.0,
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
    old_features, old_coin_distances = state_to_features(self, old_game_state)
    new_features, new_coin_distances = state_to_features(self, new_game_state)

    # Custom events to hand out reward
    # event for moving closer to closest coin / farther away
    if e.COIN_COLLECTED not in events:
        if new_coin_distances[0] < old_coin_distances[0]:
            events.append(MOVE_CLOSER_TO_COIN)
        elif new_coin_distances[0] > old_coin_distances[0]:
            events.append(MOVE_AWAY_FROM_COIN)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_features, self_action, new_features, reward_from_events(self, events)))


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
    # TODO: Maybe end of game events?
    # Update the Q Values
    last_features, _ = state_to_features(self, last_game_state)
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
    return reward_sum
