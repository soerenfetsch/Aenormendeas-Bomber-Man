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
        new_key_000 = key[:6] + [-1] + key[6:] + [0] + [0]
        new_key_100 = key[:6] + [0] + key[6:] + [0] + [0]
        new_key_200 = key[:6] + [1] + key[6:] + [0] + [0]
        new_key_300 = key[:6] + [2] + key[6:] + [0] + [0]
        new_key_400 = key[:6] + [3] + key[6:] + [0] + [0]

        new_key_010 = key[:6] + [-1] + key[6:] + [1] + [0]
        new_key_110 = key[:6] + [0] + key[6:] + [1] + [0]
        new_key_210 = key[:6] + [1] + key[6:] + [1] + [0]
        new_key_310 = key[:6] + [2] + key[6:] + [1] + [0]
        new_key_410 = key[:6] + [3] + key[6:] + [1] + [0]

        new_key_001 = key[:6] + [-1] + key[6:] + [0] + [1]
        new_key_101 = key[:6] + [0] + key[6:] + [0] + [1]
        new_key_201 = key[:6] + [1] + key[6:] + [0] + [1]
        new_key_301 = key[:6] + [2] + key[6:] + [0] + [1]
        new_key_401 = key[:6] + [3] + key[6:] + [0] + [1]

        new_key_011 = key[:6] + [-1] + key[6:] + [1] + [1]
        new_key_111 = key[:6] + [0] + key[6:] + [1] + [1]
        new_key_211 = key[:6] + [1] + key[6:] + [1] + [1]
        new_key_311 = key[:6] + [2] + key[6:] + [1] + [1]
        new_key_411 = key[:6] + [3] + key[6:] + [1] + [1]

        new_keys = (new_key_000,
                    new_key_001,
                    new_key_100,
                    new_key_101,
                    new_key_200,
                    new_key_201,
                    new_key_300,
                    new_key_301,
                    new_key_400,
                    new_key_401,
                    new_key_010,
                    new_key_011,
                    new_key_110,
                    new_key_111,
                    new_key_210,
                    new_key_211,
                    new_key_310,
                    new_key_311,
                    new_key_410,
                    new_key_411)
        
        for new_key in new_keys:
            new_key = tuple(new_key)
            new_q_table[new_key] = val

    # Save the new Q-Table as pkl file
    with open(NEW_Q_TABLE_FILE, "wb") as file:
        pickle.dump(new_q_table, file)

    # print("Old Table:", q_table)

    # print("New Table:", new_q_table)