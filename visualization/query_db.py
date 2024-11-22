import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
from nltk.tokenize import RegexpTokenizer
from pymongo import MongoClient

import study_info

# data base stuff
MONGO_URL = "INSERT"
MYCLIENT = MongoClient(MONGO_URL)
MYDB = MYCLIENT["TangramsMultiRef"]
GAME_COLLECTION = MYDB["games"]
ROUND_COLLECTION = MYDB["rounds"]
PLAYERS_COLLECTION = MYDB["players"]
TREATMENTS_COLLECTION = MYDB["treatments"]

# directory things
PICKLE_DIR = "pickle_files/"
GAME_DIR = PICKLE_DIR + "games/"
PLAYER_DIR = PICKLE_DIR + "players/"


ANNOTATIONS = [
    "",
    "sep_12_24",
]
SURVEY_RESPONSES = [
    "",
    "veryDissatisfied",
    "dissatisfied",
    "somewhatDissatisfied",
    "somewhatSatisfied",
    "satisfied",
    "verySatisfied",
]


def svg2png(lst):
    return [t[:-3] + "png" for t in lst]


@st.cache_data
def get_all_games():
    """
    Returns a dictionary that maps annotation to a dictionary of all the games
    with that annotation
    """
    ret = {}
    for annotation in ANNOTATIONS[1:]:
        annos_file = GAME_DIR + annotation

        if os.path.isfile(annos_file):
            with open(annos_file, "rb") as file:
                ret[annotation] = pickle.load(file)
        else:
            ret[annotation] = get_all_annotated_games(annotation)
            with open(annos_file, "wb") as file:
                pickle.dump(ret[annotation], file)
    all_games_dict = {}
    for annos in ret:
        all_games_dict.update(ret[annos])
    ret["all"] = all_games_dict
    return ret


@st.cache_data
def get_all_players():
    """
    Returns a dictionary that maps annotation to a dictionary of all the players
    who played games with that annotation
    """
    ret = {}
    if len(os.listdir(PLAYER_DIR)) != len(ANNOTATIONS) - 1:
        rewrite = True
    else:
        rewrite = False

    for idx in range(1, len(ANNOTATIONS)):
        annotation = ANNOTATIONS[idx]
        annos_file = PLAYER_DIR + annotation
        if os.path.isfile(annos_file):
            with open(annos_file, "rb") as file:
                ret[annotation] = pickle.load(file)
        else:
            ret[annotation] = get_all_annotated_players(annotation)
            with open(annos_file, "wb") as file:
                pickle.dump(ret[annotation], file)

    combined = {}
    for annos in ret:
        combined = combine_player_dicts(combined, ret[annos])
    ret["all"] = combined
    if rewrite:
        study_info.get_study_info()
    return ret


def get_all_annotated_games(annotation):
    """
    Returns all of the stats we want for every game in [annotation]

    This function queries the database
    """
    query = {"data.annotation": annotation}
    result = GAME_COLLECTION.find(query)

    game_dict = {}

    def per_doc_fn(doc):
        _id = doc["_id"]
        game_dict_id = dict()
        treatment_id = doc["treatmentId"]
        treatment_name = TREATMENTS_COLLECTION.find({"_id": treatment_id})[0]["name"]
        rnd = ROUND_COLLECTION.find({"gameId": _id})[0]
        time = np.sum(rnd["data"]["timeList"])  # in seconds
        game_length = str(round(time / 60, 3)) + " mins "
        data = rnd["data"]

        if (
            "endReason" in doc.keys()
            and doc["endReason"] in ("failed", "cancelled", "finished_early")
        ) or doc["status"] in ("failed", "cancelled", "finished_early"):
            print("game skipped: " + _id)
            return None
        elif doc["status"] == "custom" or (
            "roundStatus" in data.keys() and "successful" not in data["roundStatus"]
        ):
            if data["blamingWho"] != "":
                # the last person to be blamed caused game to end
                reason = data["blamingWho"] + " idled"
            else:
                reason = data["roundStatus"] + " idled"
        else:
            if len(data["target"]) == data["rightCount"] and data["wrongCount"] == 0:
                reason = "successful"
            elif set(data["target"]) == set(data["clicks"]):
                reason = "successful"
            else:
                reason = "unsuccessful"

        speaker = data["speaker"]["urlParams"]["workerId"]
        try:
            listener = data["listener"]["urlParams"]["workerId"]
            has_bot_listener = False
        except KeyError:
            listener = data["listener"]["bot"]
            has_bot_listener = True

        tokenizer = RegexpTokenizer(r"\w+")
        lst = [len(tokenizer.tokenize(msg["text"])) for msg in rnd["data"]["chat"]]
        msg_len = round(np.mean(lst), 3) if lst != [] else "N/A"

        # get chat:
        chat = []
        c = 0
        turnCount = len(data["turnList"])
        for i in range(turnCount):
            if data["turnList"][i]["secUntilSend"] == -1:
                chat.append("")
            else:
                chat.append(data["chat"][c]["text"])
                c += 1

        # extract split from config file name
        split = None
        for s in ("train", "dev", "test"):
            # for example: public/games/split_train/game_json_15527.json
            if s in doc["data"]["configFile"]:
                split = s
                break
        assert split is not None
        config = doc["data"]["configFile"]

        game_dict_id = {
            "_id": _id,
            "annotation": annotation,
            "len": game_length,
            "turn": turnCount,
            "end": reason,
            "speaker": speaker,
            "listener": listener,
            "msg": msg_len,
            # get image display related stuff
            "targets": svg2png(rnd["data"]["target"]),
            "context": svg2png(
                [x["path"] for x in data["tangrams"][0]]
            ),  # speaker's view
            "context_listener": svg2png(
                [x["path"] for x in data["tangrams"][1]]
            ),  # listener's view
            "clicks": [svg2png(x["clicks"]) for x in data["turnList"]],
            "chat": chat,
            "turnList": data["turnList"],
            "split": split,
            "config": config,
            "treatment": treatment_name,
            "created_at": rnd["createdAt"].strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": rnd["updatedAt"].strftime("%Y-%m-%d %H:%M:%S"),
        }

        # compute bonus and look at ratings
        gamers = PLAYERS_COLLECTION.find({"gameId": _id})
        count = 0
        for _ in gamers:
            count += 1
        if count != 2:
            print(count)
            print("game id: " + _id)
            return None
        # call again because cursor is consumed (i think)
        gamers = PLAYERS_COLLECTION.find({"gameId": _id})
        for gamer in gamers:  # only 2 gamers
            is_bot_listener = "bot" in gamer
            gamer_data = gamer["data"]
            role = gamer_data["role"]
            bonus = 0.0 if is_bot_listener else gamer_data["bonus"]
            hrly = round(bonus / time * 60 * 60, 2)
            if "ExitSurvey" in gamer["exitStepsDone"]:
                if "surveyResponses" in gamer_data.keys():
                    # players filled out the survey
                    sur_res = gamer_data["surveyResponses"]
                    if role == "speaker":
                        # represents how the speaker feels about their partner
                        if annotation == "jul_10":
                            game_dict_id["speaker_sat"] = SURVEY_RESPONSES.index(
                                sur_res["satisfied"]
                            )
                        else:
                            game_dict_id["speaker_sat"] = sur_res["satisfied"]
                        game_dict_id["comp"] = int(sur_res["comprehension"])
                        if "comments" in sur_res.keys():
                            game_dict_id["s comments"] = sur_res["comments"]
                        else:
                            game_dict_id["s comments"] = "N/A"
                    else:  # player is the listener
                        if annotation == "jul_10":
                            game_dict_id["listener_sat"] = SURVEY_RESPONSES.index(
                                sur_res["satisfied"]
                            )
                        else:
                            game_dict_id["listener_sat"] = sur_res["satisfied"]
                        game_dict_id["gram"] = sur_res["grammatical"]
                        game_dict_id["ambi"] = sur_res["ambiguous"]
                        game_dict_id["clear"] = sur_res["clear"]
                        if "comments" in sur_res.keys():
                            game_dict_id["l comments"] = sur_res["comments"]
                        else:
                            game_dict_id["l comments"] = "N/A"
                else:  # someone idled
                    feedback = gamer_data["errorSurveyResponses"]["feedback"]
                    if role == "speaker":
                        # listener is the one who idled
                        game_dict_id["s comments"] = (
                            feedback if feedback != "" else "N/A"
                        )
                        game_dict_id["l comments"] = "idled"
                    else:
                        # speaker who idled
                        game_dict_id["l comments"] = (
                            feedback if feedback != "" else "N/A"
                        )
                        game_dict_id["s comments"] = "idled"
            else:
                # no exit survey was completed
                if role == "listener":
                    # just set everything to N/A and deal with it later
                    game_dict_id["listener_sat"] = "N/A"
                    game_dict_id["gram"] = "N/A"
                    game_dict_id["ambi"] = "N/A"
                    game_dict_id["clear"] = "N/A"
                    game_dict_id["l comments"] = "N/A"
                else:
                    game_dict_id["speaker_sat"] = "N/A"
                    game_dict_id["comp"] = "N/A"
                    game_dict_id["s comments"] = "N/A"

            if role == "listener":
                if annotation != "jul_10":
                    game_dict_id["l lobby"] = round(
                        gamer_data["lobbyWaitTime"] / 60, 2
                    )
                    total_time = time + gamer_data["lobbyWaitTime"]
                    hrly = round(bonus / total_time * 60 * 60, 2)
                    game_dict_id["exp_l"] = (
                        (not is_bot_listener)
                        and gamer_data["expert"]
                        and gamer_data["addPoints"] == 0
                    )
                game_dict_id["listener_hrly"] = hrly
                game_dict_id['l ip'] = gamer_data.get('hashedIP', 'N/A')
            else:
                if annotation != "jul_10":
                    game_dict_id["s lobby"] = round(
                        gamer_data["lobbyWaitTime"] / 60, 2
                    )
                    total_time = time + gamer_data["lobbyWaitTime"]
                    hrly = round(bonus / total_time * 60 * 60, 2)
                    game_dict_id["exp_s"] = (
                        gamer_data["expert"] and gamer_data["addPoints"] == 0
                    )
                game_dict_id["speaker_hrly"] = hrly
                game_dict_id["s ip"] = gamer_data.get('hashedIP', 'N/A')
        # checking type of game
        if annotation == "jul_10":
            game_dict_id["type"] = "NN"  # at jul_10, all games were Novice novice
        else:
            try:
                if game_dict_id["exp_s"] and game_dict_id["exp_l"]:
                    game_dict_id["type"] = "EE"
                elif game_dict_id["exp_s"] or game_dict_id["exp_l"]:
                    game_dict_id["type"] = "EN"
                else:
                    game_dict_id["type"] = "NN"
                if has_bot_listener:
                    game_dict_id["type"] = game_dict_id["type"][0] + "B"
            except:
                print("in try except")
                print(_id)
                print(game_dict_id["end"])
                if game_dict_id["exp_s"] and game_dict_id["exp_l"]:
                    game_dict_id["type"] = "EE"
                elif game_dict_id["exp_s"] or game_dict_id["exp_l"]:
                    game_dict_id["type"] = "EN"
                else:
                    game_dict_id["type"] = "NN"
        return game_dict_id

    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(per_doc_fn, result))
        results = [r for r in results if r is not None]
        game_dict = {r["_id"]: r for r in results}
    return game_dict


def get_all_annotated_players(annotation):
    """
    Gets all the players that played in games with [annotation]
    if [annotation] is "all", we get every player

    This function queries the database
    """
    fetch_all_games = get_all_games()  # dictionary that maps annotations to games
    all_games = fetch_all_games[annotation]

    player_dict = {}
    for game in all_games.keys():
        res = PLAYERS_COLLECTION.find({"gameId": game})
        for r in res:
            try:
                playerId = r["urlParams"]["workerId"]
            except KeyError:
                playerId = r["bot"]
            if playerId not in player_dict:
                # create a set of studies this player has played in
                studies = set((all_games[game]["annotation"],))
                player_dict[playerId] = {"studs": studies}
            else:
                # add to the set of studies participated in
                player_dict[playerId]["studs"].add(all_games[game]["annotation"])

    def per_player_fn(player):
        res = PLAYERS_COLLECTION.find(
            {"urlParams.workerId": player}
        )  # TODO: bots do not have a player entry

        # find all game_ids useful
        def filter_relevant_game_id_and_role(game):
            if "gameId" not in game.keys():
                return None
            game_id = game["gameId"]
            gamers = PLAYERS_COLLECTION.find({"gameId": game_id})
            gamers = [g for g in gamers]
            if len(gamers) != 2:
                return None
            if game_id not in all_games.keys():
                return None
            return game_id, game

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(filter_relevant_game_id_and_role, res))
            results = [r for r in results if r is not None]

        total_games = 0
        successful_games = 0
        idled_games = 0
        hrly = []
        msg_len = []
        sat_rating = []
        promo_points = 0
        expert = False
        status = "Novice"
        for game_id, game in results:
            total_games += 1
            role = game["data"]["role"]
            if "idled" in all_games[game_id]["end"]:
                if role in all_games[game_id]["end"]:
                    idled_games += 1
            else:  # game was either successful or unsuccessful
                if all_games[game_id]["end"] == "successful":
                    successful_games += 1
                if role == "speaker":
                    if all_games[game_id]["listener_sat"] != "N/A":
                        sat_rating.append(
                            float(all_games[game_id]["listener_sat"])
                        )
                    else:
                        print(game_id)
                else:
                    if all_games[game_id]["speaker_sat"] != "N/A":
                        sat_rating.append(
                            float(all_games[game_id]["speaker_sat"])
                        )
                    else:
                        print(game_id)

            if role == "speaker":
                hrly.append(all_games[game_id]["speaker_hrly"])
            else:
                hrly.append(all_games[game_id]["listener_hrly"])

            if role == "speaker" and all_games[game_id]["msg"] != "N/A":
                msg_len.append(float(all_games[game_id]["msg"]))

            if annotation == "all":
                if (
                    "isLastGame" in game["data"].keys()
                    and game["data"]["isLastGame"]
                ):
                    # only check points on the last game
                    promo_points = game["data"]["points"]
                    if expert or game["data"]["expert"]:
                        promo_points = "Expert"
                else:
                    if "isLastGame" not in game["data"].keys():
                        # didn't play past jul_10
                        promo_points = "N/A"
            elif annotation != "jul_10":
                if (
                    promo_points != "Expert"
                    and promo_points < game["data"]["points"]
                ):
                    promo_points = game["data"]["points"]
                if expert or game["data"]["expert"]:
                    promo_points = "Expert"
            else:
                promo_points = "N/A"
            if promo_points == "Expert":
                status = "Expert"

        hourly = round(np.mean(hrly), 2) if len(hrly) != 0 else "N/A"
        msg = round(np.mean(msg_len), 2) if len(msg_len) != 0 else "N/A"
        sat = round(np.mean(sat_rating), 2) if len(sat_rating) != 0 else "N/A"
        succ_rate = 0 if total_games == 0 else round(successful_games / total_games, 4)
        idle_rate = 0 if total_games == 0 else round(idled_games / total_games, 4)
        ret = {
            "player": player,
            "status": status,
            "total": total_games,
            "succ_games": successful_games,
            "succ_rate": succ_rate,
            "idle_games": idled_games,
            "idle_rate": idle_rate,
            "hrly": hourly,
            "msg": msg,
            "rating": sat,
            "promo_points": promo_points,
        }
        return ret

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(per_player_fn, player_dict))
        for r in results:
            player = r.pop("player")
            player_dict[player].update(r)

    return player_dict


def combine_player_dicts(dict1, dict2):
    """
    Combines dict1 and dict2 so that players in the overlap get new stats reflecting
    both dictionaries
    """
    ret = copy.deepcopy(dict1)
    for player in dict2.keys():
        if player not in ret:
            ret[player] = dict2[player]
        else:
            if player == "4ae59479481409c5c6e34df2a66b9ae5":
                print(ret[player])
                print(dict2[player])
            p = ret[player]
            o = dict2[player]
            total = p["total"] + o["total"]
            succ = p["succ_games"] + o["succ_games"]
            idled = p["idle_games"] + o["idle_games"]

            studs = p["studs"].union(o["studs"])

            # promo points
            if p["promo_points"] == "Expert" or o["promo_points"] == "Expert":
                promo_points = "Expert"
            else:
                if p["promo_points"] == "N/A":
                    promo_points = o["promo_points"]
                elif dict2[player]["promo_points"] == "N/A":
                    promo_points = p["promo_points"]
                else:
                    promo_points = max(p["promo_points"], o["promo_points"])

            # combine hourly
            hourly = combine_stats(p, o, "hrly", total)

            # combine msg
            msg = combine_stats(p, o, "msg", total)
            msg = round(msg, 2) if msg != "N/A" else "N/A"

            # combine sat
            sat = combine_stats(p, o, "rating", total)
            sat = sat if sat == "N/A" else round(sat, 2)
            try:
                ret[player].update(
                    {
                        "studs": studs,
                        "total": total,
                        "succ_games": succ,
                        "succ_rate": round(succ / total, 4),
                        "idle_games": idled,
                        "idle_rate": round(idled / total, 4),
                        "hrly": round(hourly, 2),
                        "msg": msg,
                        "rating": sat,
                        "promo_points": promo_points,
                    }
                )
            except ZeroDivisionError:
                if total == 0:
                    print(f"{player=} played 0 games")

    return ret


def combine_stats(player, other, stat, total):
    if player[stat] == "N/A" and other[stat] == "N/A":
        return "N/A"
    elif player[stat] == "N/A":
        return other[stat] * other["total"] / total
    elif other[stat] == "N/A":
        return player[stat] * player["total"] / total
    else:
        return (player[stat] * player["total"] + other[stat] * other["total"]) / total


if __name__ == "__main__":
    get_all_games()
    get_all_players()
