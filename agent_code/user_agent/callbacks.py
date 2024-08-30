import os
import sys
DIR = os.path.dirname(__file__)
AGENT_DIR = os.path.join(DIR, "..")
sys.path.append(AGENT_DIR)

from aenormendeas_agent.callbacks import state_to_features



def setup(self):
    pass


def act(self, game_state: dict):
    features, _ = state_to_features(self, game_state)
    print(f"Given gamestate bombs: {game_state['bombs']}")
    print(f"Given explosion field: {game_state['explosion_map']}")
    print(f"Given features: {features}")
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
