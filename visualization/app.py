"""
A simple way to visualize games

Written with help from a cheat sheet created by
    @daniellewisDL : https://github.com/daniellewisDL
"""

import base64
import csv
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import query_db
import streamlit as st
from PIL import Image, ImageOps
from pymongo import MongoClient

IMAGE_DIR = "../data/tangram_pngs"


st.set_page_config(
    page_title="Multi-reference Game Visualizations",
    page_icon="tangram-human-color.png",
    layout="wide",
    initial_sidebar_state="auto",
)


### Utility ###


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def make_cols(col_list, col_titles, left_orient, txt_size):
    """
    Gives the columns in col_list titles according to col_titles.
    The titles in left_orient will be left oriented
    """
    for i in range(len(col_list)):
        with col_list[i]:
            if col_titles[i] in left_orient:
                st.write(
                    f'<div style="display: flex; justify-content: left; ">'
                    f'<span style="font-size:{txt_size}px;font-weight:bold">{col_titles[i]}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    f'<div style="display: flex; justify-content: center; ">'
                    f'<span style="font-size:{txt_size}px;font-weight:bold">{col_titles[i]}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )


def col_write(
    col,
    content,
    display="flex",
    orient="center",
    txt_size=14,
    color=0,
):
    """
    Uses markdown to write lines, specifically in columns, according to params
    """
    col.write(
        f'<div style="display: {display}; justify-content: {orient}; ">'
        f'<span style="font-size:{txt_size}px; color:{color}">{content}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def color_code_reason(reason, cols, reason_dict):
    if reason == "successful":
        reason_dict["succ"] += 1
        col_write(cols, reason, color="green")
    elif reason == "unsuccessful":
        reason_dict["unsucc"] += 1
        col_write(cols, reason, color="red")
    elif reason == "speaker idled":
        reason_dict["s_i"] += 1
        col_write(cols, reason, color="#0067C7")
    else:
        reason_dict["l_i"] += 1
        col_write(cols, reason, color="#141483")


def summarize_reason(cols, reason_dict):
    col_write(cols, "Success: " + str(reason_dict["succ"]), color="green")
    col_write(cols, "Unsuccess: " + str(reason_dict["unsucc"]), color="red")
    col_write(cols, "Speaker Idled: " + str(reason_dict["s_i"]), color="#0067C7")
    col_write(cols, "Listener Idled: " + str(reason_dict["l_i"]), color="#141483")


def color_code_type(g_type, col, type_dict):
    if g_type == "EE":
        type_dict["ee"] += 1
        col_write(col, g_type, color="#007F68")
    elif g_type == "EN":
        type_dict["en"] += 1
        col_write(col, g_type, color="#B46800")
    elif g_type == "NN":
        type_dict["nn"] += 1
        col_write(col, g_type, color="#FF2C2C")
    elif g_type == "NB":
        type_dict["nb"] += 1
        col_write(col, g_type, color="#B46800")
    elif g_type == "EB":
        type_dict["eb"] += 1
        col_write(col, g_type, color="#B46800")
    else:
        raise NotImplementedError(f"unknown type {g_type=}")



def summarize_type(col, type_dict):
    col_write(col, "EE: " + str(type_dict["ee"]), color="#007F68")
    col_write(col, "EN: " + str(type_dict["en"]), color="#B46800")
    col_write(col, "NN: " + str(type_dict["nn"]), color="#FF2C2C")
    col_write(col, "NB: " + str(type_dict["nb"]), color="#B46800")
    col_write(col, "EB: " + str(type_dict["eb"]), color="#B46800")


def markdown_write(text, txt_size, line_height=2, color=0):
    st.write(
        f'<span style="font-size:{txt_size}px; line-height:{line_height}; color:{color}">{text}',
        unsafe_allow_html=True,
    )


def add_new_states():
    """
    Adds display, game_info, justClicked, player_id, history, and home into state
    display is for display purposes (which screens to display)
    game_info, player_id, and game_id are for figuring out what to display
    justClicked: tbh not sure why it's needed, but without it, clicking on game buttons from game summary doesn't work
    home: keeps track of what the "Back to home" button brings you back to.
    """
    if "display" not in st.session_state:
        st.session_state.display = "intro"
    if "game_info" not in st.session_state:
        st.session_state.game_info = {}
    if "justClicked" not in st.session_state:
        st.session_state.justClicked = False
    if "player_id" not in st.session_state:
        st.session_state.player_id = ""
    if "history" not in st.session_state:
        st.session_state.history = []  # start as an empty list
    if "home" not in st.session_state:
        st.session_state.home = "intro"


def make_buttons(show_back=True):
    """
    Makes the two buttons at the top of the screen.
    [show_back] is only false when the input player/game doesn't exist
    """
    cols = st.columns([1, 11, 2], gap="small")
    if show_back:
        cols[0].button("Back", on_click=go_back, key="back1")
    if st.session_state.home == "game":
        cols[2].button("Back to home", on_click=go_game_home, key="go game home")
    elif st.session_state.home == "player":
        cols[2].button("Back to home", on_click=go_player_home, key="go player home")


### Get from MONGODB ###


def get_games(filter):
    """
    Returns a nested dictionary containing game ids as keys according filter
    The contents of filter are:
    1. annotation
    2. message length
    3. game status

    """
    anno = filter["annotation"]
    fetched_games = query_db.get_all_games()  # all of the games ever
    all_games = fetched_games[anno]

    game_dict = {}
    player = filter["player_id"]

    for _id, info in all_games.items():
        try:
            if (
                (info["end"] in filter["status"])
                and (
                    info["msg"] == "N/A"
                    or (filter["msglen"][0] < info["msg"] < filter["msglen"][1])
                )
                and (
                    player == ""  # not asking for a player
                    or player == info["speaker"]
                    or player == info["listener"]
                )
                and (info["type"] in filter["type"])
            ):
                game_dict[_id] = info
        except:
            print(info)
            print(_id)

    return game_dict


def get_players(filter):
    """
    Returns a nested dictionary of players according to filter:
    filtering based on:
    1. success rate
    2. satisfaction
    3. average msg length
    """
    all_players = query_db.get_all_players()
    annos = filter["annotation"]
    all_players = all_players[annos]

    players = {}

    for _id, info in all_players.items():
        if (
            (filter["succ_rate"][0] <= info["succ_rate"] <= filter["succ_rate"][1])
            and (
                info["rating"] == "N/A"
                or (filter["rating"][0] <= info["rating"] <= filter["rating"][1])
            )
            and (
                info["msg"] == "N/A"
                or (filter["msglen"][0] <= info["msg"] <= filter["msglen"][1])
            )
            and ("status" not in info or info["status"] in filter["rank"])
        ):
            players[_id] = info
    return players


def get_game(filter):
    """
    Returns a dictionary containing information of just one ;
    uses filter for the sake of consistency

    Precondition: filter must contain key "game_id"
    """
    game_id = filter["game_id"]
    fetched_games = query_db.get_all_games()
    all_games = {}
    for annos in fetched_games:
        all_games.update(fetched_games[annos])

    try:
        return all_games[game_id]
    except:
        return ""


### Set ###


def set_player(player_id, curr):
    """
    Brings you to the front page with all games of this player

    curr describes the current state; generally either player summary or a specific game
    """
    st.session_state.history.append(curr)
    st.session_state.game_id = ""
    st.session_state.player_id = player_id
    st.session_state.id = ""
    st.session_state.display = "spec_player"


### Button Control Flow ###
# idea: every time you click a non-back button, you specify where you came from in a list:
# ["display_type", "id , else None"]
# This gets added to a list that back buttons then use to return to things


def view_game(game_id, game_info, curr=["game_sum"]):
    """
    Button callback for viewing a game with id [game_id]
    curr denotes what is currently displayed, and curr will be saved to history
     - defaults to game summary
     - if a player, then the first element is "spec_player", and the second element is the player's ID
     - if a game, then the first element is "spec_game", followed by [game_id, game_info]

    Precondition: curr must be a non-empty list
    """
    st.session_state.display = "spec_game"
    st.session_state.game_id = game_id
    st.session_state.game_info = game_info
    st.session_state.justClicked = True
    st.session_state.history.append(curr)


def go_back():
    """
    Handles going back whenever called (only called by a button)
    """
    # remove the last element of history and return it
    back_one = st.session_state.history.pop()

    if back_one[0] == "game_sum":
        go_game_home()
    elif back_one[0] == "spec_game":
        # just came from a specific game
        st.session_state.display = "spec_game"
        st.session_state.game_id = back_one[1]
        st.session_state.game_info = back_one[2]
        st.session_state.justClicked = True
    elif back_one[0] == "spec_player":
        st.session_state.game_id = ""
        st.session_state.player_id = back_one[1]
        st.session_state.id = ""
        st.session_state.display = "spec_player"
    elif back_one[0] == "player_sum":
        go_player_home()


def go_game_home():
    """
    Sets display to game summary; resets everything else
    """
    st.session_state.game_id = ""
    st.session_state.game_info = {}
    st.session_state.id = ""
    st.session_state.player_id = ""
    st.session_state.display = "game_sum"
    st.session_state.history = []  # clear history


def go_player_home():
    """
    Sets display to player summary; resets everything else
    """
    st.session_state.game_id = ""
    st.session_state.game_info = {}
    st.session_state.id = ""
    st.session_state.player_id = ""
    st.session_state.display = "player_sum"
    st.session_state.history = []  # clear history


### Display ###


def display_no_game():
    make_buttons(False)
    st.title("This game doesn't exist.")
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)


def display_no_player():
    make_buttons(False)
    st.title("This player doesn't exist.")
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)


def display_title():
    """
    Displays the title screen
    """
    st.session_state.history = []  # resets history
    st.title("Please select a game annotation")
    query_db.get_all_games()
    query_db.get_all_players()
    image1 = Image.open("tangram-human-color.png")
    image1 = image1.resize((50, 50))
    st.image([image1] * 13)


def display_game_summary(filter):
    """
    Displays the games according the filter, which is a dictionary
    Returns the game id of a game we want to see
    """
    st.title("Game Summary")
    games = get_games(filter)
    st.write("Total number of games: " + str(len(games.keys())))
    annotation = filter["annotation"]
    if annotation != "all":
        with open("pickle_files/study_info", "rb") as f:
            study_info = pickle.load(f)
            st.write("Total players: " + str(study_info[annotation]["total players"]))
            st.write("New players: " + str(study_info[annotation]["new players"]))
            st.write(
                "Returning players: " + str(study_info[annotation]["returning players"])
            )

    # create columns:

    cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    header_lst = [
        "Game Ids",
        "Length",
        "#Turns",
        "Ending",
        "Speaker",
        "Listener",
        "Msg Len",
        "Type",
        "Treatment",
    ]
    make_cols(cols, header_lst, ["Game Ids", "Speaker", "Listener"], 28)

    lists = {"len": 0, "turns": 0, "msg": []}

    reason_dict = {"succ": 0, "unsucc": 0, "s_i": 0, "l_i": 0}
    type_dict = {
        "ee": 0,
        "en": 0,
        "nn": 0,
        "nb": 0,
        "eb": 0,
    }
    tot = 0
    for _id, info in games.items():
        tot += 1
        # new columns for alignment purposes
        cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1, 1], gap="medium")
        reason = info["end"]
        speaker = info["speaker"]
        listener = info["listener"]
        leng = info["len"]
        turns = info["turn"]

        # current would be game_sum (default):
        cols[0].button(_id, on_click=view_game, args=[_id, info])

        # updating the rest of the columns
        col_write(cols[1], leng)
        col_write(cols[2], turns)

        # color-coding reason
        color_code_reason(reason, cols[3], reason_dict)

        cols[4].text(speaker)
        cols[5].text(listener)
        col_write(cols[6], info["msg"])

        color_code_type(info["type"], cols[7], type_dict)
        col_write(cols[8], info["treatment"])

        lists["len"] = (lists["len"] * (tot - 1) + (float(leng[:-6]))) / tot
        lists["turns"] = (lists["turns"] * (tot - 1) + turns) / tot
        if info["msg"] != "N/A":
            # need list bc of N/A
            lists["msg"].append(int(info["msg"]))

    # compute averages & record
    cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    avg_len = round(lists["len"], 3)
    avg_turn = round(lists["turns"], 3)
    avg_msg = round(np.mean(lists["msg"]), 3)
    col_write(
        cols[0],
        "Average (" + str(tot) + ")",
        txt_size=18,
        orient="left",
    )
    col_write(cols[1], str(avg_len) + " mins")
    col_write(cols[2], str(avg_turn) + " turns")
    summarize_reason(cols[3], reason_dict)
    col_write(cols[6], str(avg_msg))
    summarize_type(cols[7], type_dict)


def display_player_summary(filter):
    """
    Displays summary of players
    """
    st.title("Player Summary")
    cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    header_lst = [
        "Player Ids",
        "Total Games",
        "Success Rate",
        "Idle Rate",
        "Avg H. Pay",
        "Avg Msg Len",
        "Avg rating",
        "Promo points",
    ]
    make_cols(cols, header_lst, ["Player Ids"], 28)

    players = get_players(filter)
    tot_play = 0
    num_experts = 0
    avg_games = 0
    tot_games = 0
    avg_succ = 0
    avg_hrly = 0
    avg_idle = 0
    for p_id, p_info in players.items():
        if p_id == "bob":
            continue
        tot_play += 1
        avg_succ = (avg_succ * (tot_play - 1) + p_info["succ_rate"]) / tot_play
        avg_games = (avg_games * (tot_play - 1) + p_info["total"]) / tot_play
        avg_idle = (avg_idle * (tot_play - 1) + p_info["idle_rate"]) / tot_play
        avg_hrly = (avg_hrly * (tot_play - 1) + p_info["hrly"]) / tot_play
        tot_games += p_info["total"]
        cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1], gap="medium")

        cols[0].button(
            p_id[:9],
            on_click=set_player,
            args=[p_id, ["player_sum"]],
            key=p_id + "player_sum",
        )
        col_write(cols[1], p_info["total"])

        col_write(cols[2], p_info["succ_rate"])
        col_write(cols[3], p_info["idle_rate"], color=0)
        col_write(cols[4], p_info["hrly"])
        col_write(cols[5], p_info["msg"])
        col_write(cols[6], p_info["rating"])
        if "promo_points" in p_info.keys():
            if p_info["promo_points"] == "Expert":
                num_experts += 1
                col_write(cols[7], p_info["promo_points"], color="green")
            else:
                col_write(cols[7], p_info["promo_points"])
        else:
            col_write(cols[7], "N/A")

    cols = st.columns(spec=[2, 1, 1, 1, 1, 1, 1, 1], gap="medium")
    col_write(
        cols[0],
        "Average (" + str(tot_play) + " players)",
        txt_size=18,
        orient="left",
    )
    col_write(cols[1], "Total: " + str(tot_games / 2) + " games")
    col_write(cols[1], "Avg: " + str(round(avg_games, 2)) + " games")
    col_write(cols[2], "Avg: " + str(round(avg_succ, 4)))
    col_write(cols[3], "Avg: " + str(round(avg_idle, 4)))
    col_write(cols[4], "Avg: $" + str(round(avg_hrly, 2)))
    col_write(cols[7], str(num_experts) + " experts")


def display_player(player, filter):
    """
    Displays player stats.
    This includes a list of games they played (and associated annotation),
    their average message length, their rating from their peers
    """
    if filter["player_id"] == "":  # from the sidebar
        player = st.session_state.player_id
        filter["player_id"] = player
    else:
        player = filter["player_id"]
    games = get_games(filter)

    if player not in query_db.get_all_players()[filter["annotation"]].keys():
        display_no_player()
        return

    make_buttons()

    st.subheader("Player: " + player)

    tab1, tab2 = st.tabs(["General", "Rating"])

    with tab1:
        # st.header("General Game Summary")
        display_gen(games, player)
    with tab2:
        display_rating(games, player)


def display_gen(games, player):
    """
    Displays the general game stats for a player, including game length, number
    of turns, how the game ended, the player's role, message length, and
    hourly play for each game in [games]
    """
    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 1])
    titles = [
        "Games Played",
        "Length",
        "Turns",
        "Ending",
        "Role",
        "Message Length",
        "Hourly Pay",
        "Type",
        "Treatment",
    ]
    make_cols(cols, titles, ["Games Played"], 28)
    reason_dict = {"succ": 0, "unsucc": 0, "s_i": 0, "l_i": 0}
    type_dict = {"ee": 0, "en": 0, "nn": 0, "eb": 0, "nb": 0}
    tot = 0
    leng = 0
    turns = 0
    msg_len = []
    hr_py = 0
    for _id, info in games.items():
        tot += 1
        cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1, 1])
        reason = info["end"]
        speaker = info["speaker"]
        role = "speaker" if player == speaker else "listener"

        if role == "speaker":
            hrly = info["speaker_hrly"] if role not in reason else 0
        else:
            hrly = info["listener_hrly"] if role not in reason else 0

        cols[0].button(
            _id,
            on_click=view_game,
            args=[_id, info, ["spec_player", player]],
            key=(_id + "playergen"),
        )
        col_write(cols[1], info["len"])
        col_write(cols[2], info["turn"])

        color_code_reason(reason, cols[3], reason_dict)

        col_write(cols[4], role)
        col_write(cols[5], info["msg"])
        col_write(cols[6], hrly)

        color_code_type(info["type"], cols[7], type_dict)
        col_write(cols[8], info["treatment"])

        leng = (leng * (tot - 1) + float(info["len"][:-5])) / tot
        turns = (turns * (tot - 1) + float(info["turn"])) / tot
        hr_py = (hr_py * (tot - 1) + hrly) / tot
        if info["msg"] != "N/A":
            msg_len.append(info["msg"])
        else:
            msg_len.append(0)

    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    col_write(cols[0], "Average (" + str(tot) + " games):", txt_size=18, orient="left")
    col_write(cols[1], str(round(leng, 3)) + " mins")
    col_write(cols[2], str(round(turns, 3)))
    summarize_reason(cols[3], reason_dict)
    col_write(cols[5], str(round(np.mean(msg_len), 4)))
    col_write(cols[6], str(round(hr_py, 2)))
    summarize_type(cols[7], type_dict)


def display_rating(games, player):
    """
    Displays [player]'s ratings from their (different) partners in the [games] played.
    """
    st.header("Player ratings")
    cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
    titles = [
        "Games Played",
        "Role",
        "Partner",
        "Satisfaction",
        "Grammar",
        "Clarity",
        "Ambiguity",
        "Comp.",
    ]
    make_cols(cols, titles, ["Games Played"], 20)

    for _id, info in games.items():
        # set columns at each iteration for alignmnet purposes
        cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
        cols[0].button(
            _id,
            on_click=view_game,
            args=[_id, info, ["spec_player", player]],
            key=(_id + "playerrate"),
        )

        role = "speaker" if player == info["speaker"] else "listener"
        col_write(cols[1], role)

        # find their partner
        p = info["listener"] if role == "speaker" else info["speaker"]
        cols[2].text(p)

        # if the game finished, give their ratings
        if role == "speaker" and "idled" not in info["end"]:
            # listener_sat reflects how the listener felt about their partner, the speaker
            col_write(cols[3], info["listener_sat"])
            col_write(cols[4], info["gram"])
            col_write(cols[5], info["clear"])
            col_write(cols[6], info["ambi"])
            col_write(cols[7], "N/A")
        elif role == "listener" and "idled" not in info["end"]:
            col_write(cols[3], info["speaker_sat"])
            for i in range(4, 7):
                col_write(cols[i], "N/A")
            col_write(cols[7], info["comp"])
        else:
            for i in range(3, len(cols)):
                col_write(cols[i], "N/A")

def display_game(id, game_info, annotation):
    """
    Displays game information for game [id]:
    game_info has the following elements:
    """
    make_buttons()
    if "exp_s" in game_info.keys() and "exp_l" in game_info.keys():
        s_exp_stat = " \u2605" if game_info["exp_s"] else ""
        l_exp_stat = " \u2605" if game_info["exp_l"] else ""
    else:
        s_exp_stat = ""
        l_exp_stat = ""

    st.header("Game ID: " + id)

    s, l = st.columns(spec=[1, 1], gap="medium")
    s.markdown("**Game Length:** " + game_info["len"])
    l.markdown("**Ending:** " + game_info["end"])
    s.text("Started At: " + game_info["created_at"])
    l.text("Ended At: " + game_info["updated_at"])
    s.button(
        "Speaker" + s_exp_stat,
        on_click=set_player,
        args=[game_info["speaker"], ["spec_game", id, game_info]],
    )
    s.text("ID: " + game_info["speaker"])
    if "s lobby" in game_info.keys():
        s.text("Lobby time: " + str(game_info["s lobby"]) + " mins")
    l.button(
        "Listener" + l_exp_stat,
        on_click=set_player,
        args=[game_info["listener"], ["spec_game", id, game_info]],
    )
    l.text("ID: " + game_info["listener"])
    if "l lobby" in game_info.keys():
        l.text("Lobby time: " + str(game_info["l lobby"]) + " mins")
    l.text("Treatment: " + game_info["treatment"])

    # start with speaker's initial view, and no clicks
    display_context(game_info["context_listener"], game_info["targets"], show_ground_truth=True)
    st.divider()

    chat_lengths = []
    turn = 0
    chat_counter = 0
    changes = []
    for turn in range(len(game_info["turnList"])):
        turn_info = game_info["turnList"][turn]
        turn_text = "Turn " + str(turn + 1)
        st.markdown("### %s" % (turn_text))

        if annotation != "jul_10":
            if turn_info["secUntilSend"] == -1:
                # speaker idled
                markdown_write("Speaker Idled", 20, color="red")
                chat_lengths.append(0)
            else:
                s, l, _ = st.columns(spec=[1, 1, 4], gap="medium")
                s.markdown(
                    "**Speaker time:** " + str(round(turn_info["secUntilSend"], 4))
                )
                if turn_info["secBetweenSendAndSubmit"] == -1:
                    # listener idled, but speaker did not idle:
                    markdown_write("Listener Idled", 20, color="red")
                else:
                    # no one idled
                    l.markdown(
                        "**Listener time:** "
                        + str(round(turn_info["secBetweenSendAndSubmit"], 3))
                    )
                # speaker sent a message:
                chat = game_info["chat"][turn]
                markdown_write(chat, 20)
        else:
            # jul_10 annotation
            if game_info["chat"][turn] == "":
                chat = "Speaker idled"
                markdown_write(chat, 20, color="red")
            else:
                chat = game_info["chat"][turn]
                markdown_write(chat, 20)
                chat_counter += 1

        if turn > 0:
            c1 = [
                img
                for img in game_info["clicks"][turn]
                if img not in game_info["clicks"][turn - 1]
            ]
            c2 = [
                img
                for img in game_info["clicks"][turn - 1]
                if img not in game_info["clicks"][turn]
            ]
            changes = c1 + c2
        else:
            changes = game_info["clicks"][0]

        display_context(
            game_info["context_listener"],
            game_info["targets"],
            changes,
            game_info["clicks"][turn],
            game_id=id,
            turn_on_vis=turn + 1,
            treatment=game_info["treatment"],
            show_ground_truth=st.session_state.show_ground_truth,
        )

        game_id = id
        turn_on_vis = turn + 1
        show_reward_select = st.session_state.annotation_mode == "rewards" and (game_id is not None) and (turn_on_vis is not None)
        if show_reward_select and turn < len(game_info["turnList"]) - 1:
            with st.form(key=f"form_{game_id}_{turn_on_vis}_reward"):
                st.markdown("Feedback")
                st.markdown(game_info['chat'][turn+1])
                reward = st.selectbox("Reward", ["neg", "neu", "pos", "na"], key=f"reward_{game_id}_{turn_on_vis}")
                submitted = st.form_submit_button("Submit", on_click=just_clicked)
                if submitted:
                    on_turn_reward_form_submit(dict(
                        reward=reward,
                        game_id=game_id,
                        turn_id=turn_on_vis,
                        treatment=game_info["treatment"]))

        turn += 1

    # do comments
    st.divider()
    markdown_write(("Speaker comments: " + game_info["s comments"]), 20)
    markdown_write(("Listener comments: " + game_info["l comments"]), 20)


def display_context(context, targets, changes=[], clicks=[], game_id=None, turn_on_vis=None, treatment=None, show_ground_truth=True):
    """
    Displays the context with targets and clicks
    """
    # first get all of the tangrams showing correctly
    tangram_list = []
    arrow = Image.open("yellow_circle.png").resize((20, 20)).convert("RGBA")
    for img in context:
        image = Image.open(os.path.join(IMAGE_DIR, img)).resize((60, 60)).convert("RGB")
        image = ImageOps.expand(image, border=2, fill="white")
        if show_ground_truth:
            if img in targets and img in clicks:  # listener selected a target image
                image = ImageOps.expand(image, border=10, fill="green")
            elif img in targets and img not in clicks:  # unselected target:
                image = ImageOps.expand(image, border=10, fill="black")
            elif img in clicks and img not in targets:  # listener selected a wrong image
                image = ImageOps.expand(image, border=10, fill="red")
            else:
                image = ImageOps.expand(image, border=10, fill="white")
        else:
            if img in clicks:  # selected post clicks this turn
                image = ImageOps.expand(image, border=10, fill="gray")
            else:
                image = ImageOps.expand(image, border=10, fill="white")
        image = ImageOps.expand(image, border=2, fill="white")
        if img in changes:
            image.paste(arrow, (68, 0), mask=arrow)
        tangram_list.append(image)


    show_checkboxes = st.session_state.annotation_mode == "clicks" and (game_id is not None) and (turn_on_vis is not None)

    if not show_checkboxes:
        st.image(tangram_list[:10])
        st.image(tangram_list[10:])
        return

    with st.form(key=f"form_{game_id}_{turn_on_vis}"):
        cols = st.columns(10, gap="small")
        checkboxes = dict()
        # NOTE: assume context size of 10
        for i in range(10):
            with cols[i]:
                key = button_key(game_id, turn_on_vis, context[i], treatment)
                st.image(tangram_list[i])
                checkboxes[key] = st.checkbox(key, key=key, label_visibility="collapsed")
        submitted = st.form_submit_button("Submit", on_click=just_clicked)
        if submitted:
            on_turn_form_submit(checkboxes)

def just_clicked():
    st.session_state.justClicked = True

def button_key(*args):
    return "+".join([str(arg) for arg in args])

def undo_button_key(key):
    return tuple(key.split("+"))

def on_turn_form_submit(checkboxes):
    labels = [undo_button_key(key) for key, checkbox in checkboxes.items() if checkbox]
    if len(labels) == 0:
        st.toast(":red[Please select at least one tangram / submit again]")
        st.session_state.justClicked = True
        return
    game_id, turn_id, _, treatment = labels[0]
    images = ", ".join([img for _, _, img, _ in labels])
    data_to_append = [game_id, turn_id, images, datetime.now()]
    csv_file = f"human_bot_annotation/{treatment}.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["game_id", "turn_on_vis", "labels", "timestamp"])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)
    st.toast(f":green[Annotations submitted for Turn {turn_id} of Game {game_id} to {treatment}.csv]", icon='🎉')
    st.session_state.justClicked = True  # to prevent the page from refreshing

def on_turn_reward_form_submit(reward_info):
    data_to_append = [
        reward_info["game_id"],
        reward_info["turn_id"],
        reward_info["reward"],
        datetime.now()]
    csv_file = f"human_bot_annotation/reward/{reward_info['treatment']}.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["game_id", "turn_on_vis", "reward", "timestamp"])
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_to_append)
    st.toast(f":green[Reward submitted for Turn {reward_info['turn_id']} of Game {reward_info['game_id']} to reward/{reward_info['treatment']}.csv]", icon='🎉')
    if reward_info['reward'] == 'neg':
        st.toast(":red[negative] feedback saved", icon='👎')
    elif reward_info['reward'] == 'pos':
        st.toast(":green[positive] feedback saved", icon='👍')
    elif reward_info['reward'] == 'neu':
        st.toast(":blue[neutral] feedback saved", icon='🤷')
    elif reward_info['reward'] == 'na':
        st.toast(":black[na] saved", icon='☠️')
    else:
        st.error("Invalid reward type")
    st.session_state.justClicked = True  # to prevent the page from refreshing

### Sidebar ###


def sidebar(filter):
    # title
    st.sidebar.markdown(
        """<img src='data:image/png;base64,{}' class='img-fluid' width=120 height=120>""".format(
            img_to_bytes("tangram-human-color.png")
        ),
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("# Multi-reference Game Visualization")

    st.sidebar.toggle("Show ground truth", value=True, key="show_ground_truth", on_change=go_game_home)

    st.sidebar.selectbox(
        "Select annotation mode",
        ["none", "clicks", "rewards"],
        key="annotation_mode",
        on_change=go_game_home,
    )

    # annotations
    filter["annotation"] = st.sidebar.radio(
        "Please select an annotation",
        [query_db.ANNOTATIONS[0]] + ["all"] + query_db.ANNOTATIONS[1:],
    )

    # Can either view a certain game or a certain player
    filter["collection"] = st.sidebar.selectbox(
        "Collection to view", ("Game", "Player")
    )

    disable_status = st.session_state.player_id == "" and filter["player_id"] == ""

    if filter["annotation"] != "":
        if filter["collection"] == "Game":
            st.session_state.home = "game"
            filter["game_id"] = st.sidebar.text_input(
                "Game Id", "", max_chars=32, key="id"
            )
            if filter["game_id"] != "":
                st.session_state.display = "spec_game"
            else:
                st.session_state.display = "game_sum"
            disable_status = False
        else:
            st.session_state.home = "player"
            filter["player_id"] = st.sidebar.text_input(
                "Player Id", "", max_chars=32, key="id"
            )
            if not disable_status:
                st.session_state.display = "spec_player"
            else:
                st.session_state.display = "player_sum"

            # message length
            filter["succ_rate"] = st.sidebar.slider(
                "Select success rate range",
                0.0,
                1.0,
                (0.0, 1.0),
                disabled=not disable_status,
            )
            filter["rating"] = st.sidebar.slider(
                "Select satisfaction range",
                0.0,
                6.0,
                (0.0, 6.0),
                disabled=not disable_status,
            )
            filter["rank"] = st.sidebar.multiselect(
                "Rank: ", ["Expert", "Novice"], ["Expert", "Novice"]
            )

        # message length
        filter["msglen"] = st.sidebar.slider(
            "Select message length range", 0.0, 35.0, (0.0, 35.0)
        )
        # game end reason, only for games
        filter["status"] = st.sidebar.multiselect(
            "End reason: ",
            ["successful", "unsuccessful", "listener idled", "speaker idled"],
            ["successful", "unsuccessful", "listener idled", "speaker idled"],
            disabled=disable_status,
        )
        filter["type"] = st.sidebar.multiselect(
            "Type: ",
            ["EE", "EN", "NN", "NB", "EB"],
            ["EE", "EN", "NN", "NB", "EB"],
            disabled=disable_status,
        )

    return filter


def main():
    """
    Runs every time a click is made
    """
    add_new_states()

    filter = {"annotation": "", "game_id": "", "player_id": ""}
    new_filter = sidebar(filter)

    if new_filter["annotation"] == "" or st.session_state.display == "intro":
        display_title()
        return None

    if new_filter["collection"] == "Game" and new_filter["game_id"] != "":
        if get_game(new_filter) == "":
            st.session_state.history.append(["game_sum"])
            display_no_game()
            return None

    if new_filter["game_id"] != "":
        st.session_state.game_id = new_filter["game_id"]
        # if we want to ga back, it would make sense to just go to summary
        st.session_state.history.append(["game_sum"])
        st.session_state.game_info = get_game(new_filter)
    elif st.session_state.justClicked:
        st.session_state.display = "spec_game"
        st.session_state.justClicked = False
    elif st.session_state.player_id != "":
        st.session_state.display = "spec_player"
    elif new_filter["player_id"] != "":
        st.session_state.player_id = new_filter["player_id"]
        st.session_state.display = "spec_player"
        st.session_state.history.append(["player_sum"])

    # (st.session_state.history)

    # display as necessary
    if st.session_state.display == "intro" or new_filter["annotation"] == "":
        display_title()
    if st.session_state.display == "game_sum" and new_filter["annotation"] != "":
        display_game_summary(new_filter)
    elif st.session_state.display == "spec_game":
        display_game(
            st.session_state.game_id,
            st.session_state.game_info,
            new_filter["annotation"],
        )
    elif st.session_state.display == "player_sum" and new_filter["annotation"] != "":
        display_player_summary(new_filter)
    elif st.session_state.display == "spec_player":
        display_player(st.session_state.player_id, new_filter)

    return None


if __name__ == "__main__":
    main()
