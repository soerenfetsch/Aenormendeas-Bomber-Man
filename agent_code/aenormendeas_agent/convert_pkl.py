import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))

Q_TABLE_FILE = os.path.join(DIR, "old_models", "coins_crates_bombs_qtable.pkl")
NEW_Q_TABLE_FILE = os.path.join(DIR, "coins_crates_bombs_players.pkl")


if __name__ == "__main__":

    assert os.path.isfile(Q_TABLE_FILE)

    # Get q table from old file
    with open(Q_TABLE_FILE, "rb") as file:
        q_table: dict = pickle.load(file)
    # q_table = {(3, 2, 3, 3,    -1, -1,         1, 1, 0, 0, 0,   0,    0): 100}

    # Transform old q table into new one

    new_q_table = {}

    # old = [3, 2, 3, 3,    -1, -1,         1, 1, 0, 0, 0,   0,    0]
    # new = [3, 2, 3, 3,    -1, -1, -1,     1, 1, 0, 0, 0,   0,    0,    0]

    for key, val in q_table.items():
        key = list(key)
        new_key_00 = key[:6] + [-1] + key[6:] + [0]
        new_key_10 = key[:6] + [0] + key[6:] + [0]
        new_key_20 = key[:6] + [1] + key[6:] + [0]
        new_key_30 = key[:6] + [2] + key[6:] + [0]
        new_key_40 = key[:6] + [3] + key[6:] + [0]

        new_key_01 = key[:6] + [-1] + key[6:] + [1]
        new_key_11 = key[:6] + [0] + key[6:] + [1]
        new_key_21 = key[:6] + [1] + key[6:] + [1]
        new_key_31 = key[:6] + [2] + key[6:] + [1]
        new_key_41 = key[:6] + [3] + key[6:] + [1]

        new_keys = (new_key_00,
                    new_key_10,
                    new_key_20,
                    new_key_30,
                    new_key_40,
                    new_key_01,
                    new_key_11,
                    new_key_21,
                    new_key_31,
                    new_key_41)
        
        for new_key in new_keys:
            new_key = tuple(new_key)
            new_q_table[new_key] = val

    # Save the new Q-Table as pkl file
    with open(NEW_Q_TABLE_FILE, "wb") as file:
        pickle.dump(new_q_table, file)

    # print("Old Table:", q_table)

    # print("New Table:", new_q_table)