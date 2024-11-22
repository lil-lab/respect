import pickle
import os
import query_db


def get_study_info():
    new_info = get_info()

    with open("pickle_files/study_info", "wb") as new:
        pickle.dump(new_info, new)


def get_info():
    """
    Iterates player dictionaries of all previous studies to generate
    study-level player information:
    outputs a dictionary that maps study date to:
    {
        'total players': xxx,
        'new players': xxx,
        'returning players': xxx
    }
    """

    # map a player to when they first joined a study
    players = set()
    player_to_date = {}
    for file_name in query_db.ANNOTATIONS:
        f = os.path.join(query_db.PLAYER_DIR, file_name)
        if os.path.isfile(f):
            with open(f, "rb") as study:
                stud = pickle.load(study)
                for player in stud.keys():
                    if player not in players:
                        k_v_pair = {player: file_name}
                        player_to_date.update(k_v_pair)
                        players.add(player)

    date_to_players = {}
    for file_name in os.listdir(query_db.PLAYER_DIR):
        f = os.path.join(query_db.PLAYER_DIR, file_name)
        if os.path.isfile(f):
            info = {"total players": 0, "new players": 0, "returning players": 0}
            with open(f, "rb") as study:
                stud = pickle.load(study)
            for player in stud.keys():
                info["total players"] += 1
                if player_to_date[player] == file_name:
                    info["new players"] += 1
                else:
                    info["returning players"] += 1
        date_to_players[file_name] = info

    print(date_to_players)
    return date_to_players


if __name__ == "__main__":
    get_study_info()
