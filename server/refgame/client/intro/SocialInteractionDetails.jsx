import React from "react";

import { Centered } from "meteor/empirica:core";
// //// Avatar stuff //////
// const names = "abcdefghijklmnopqrstuvwxyz".toUpperCase().split(""); //for the players names (we will call them A, B, C etc)
const names = ["Blue", "Green", "Pink", "Yellow"]; // for the players names to match avatar color
const avatarNames = ["Colton", "Aaron", "Alex", "Tristan"]; // to do more go to https://jdenticon.com/#icon-D3
const nameColor = ["#3D50B7", "#70A945", "#DE8AAB", "A59144"]; // similar to the color of the avatar

export default class SocialInteractionDetails extends React.Component {
  state = {
    satisfied: false
  };

  renderPlayer(player, self = false) {
    return (
      <div className="player" key={player._id}>
        <span className="image">
          <span
            className={`satisfied bp3-tag bp3-round ${player.satisfied ? "bp3-intent-success" : "bp3-intent-danger"
              }`}
          >
            <span
              className={`bp3-icon-standard ${player.satisfied ? "bp3-icon-tick" : "bp3-icon-cross"
                }`}
            />
          </span>

          <img src={player.avatar} />
        </span>
        {/* <span className="name" style={{ color: player.get("nameColor") }}> */}
        <span className="name" style={{ color: player.nameColor }}>
          {player.name}
          {self ? " (You)" : ""}
        </span>
      </div>
    );
  }

  handleSatisfaction = (satisfied, event) => {
    event.preventDefault();
    this.setState({ satisfied: satisfied });
  };

  render() {
    const { hasPrev, hasNext, onNext, onPrev, treatment } = this.props;
    const player = {
      _id: 0,
      name: names[0],
      nameColor: nameColor[0],
      avatar: `/avatars/jdenticon/${avatarNames[0]}`,
      satisfied: this.state.satisfied
    };

    const otherPlayers = [
      {
        _id: 1,
        name: names[1],
        nameColor: nameColor[1],
        avatar: `/avatars/jdenticon/${avatarNames[1]}`,
        satisfied: false
      },
      {
        _id: 2,
        name: names[2],
        nameColor: nameColor[2],
        avatar: `/avatars/jdenticon/${avatarNames[2]}`,
        satisfied: true
      }
    ];
    return (
      <Centered>
        <div className="instructions">
          <h1 className={"bp3-heading"}> In-Game Chat</h1>
          <p>
            You may communicate with your teammates through the in-game
            chat. Whatever you write will appear to your partner. {" "}
            You can use the chat function however you like, but please note that
            <strong>the speaker must send a message before the listener is allowed to make a selection</strong>.
          </p>

          <p>
            You and your teammates have{" "}
            {(treatment.turnTime)} seconds on each round to
            select a tangram. If you do not select a tangram in this window, you will automatically{" "}
            <strong>progress to the next task when the time is up</strong> and will not recieve a bonus,
            so please stay focused.
            If your pair selects a picture before the other pairs are finished, please wait and
            for the other pair to complete their selections.
          </p>

          <button
            type="button"
            className="bp3-button bp3-intent-nope bp3-icon-double-chevron-left"
            onClick={onPrev}
            disabled={!hasPrev}
          >
            Previous
          </button>
          <button
            type="button"
            className="bp3-button bp3-intent-primary"
            onClick={onNext}
            disabled={!hasNext}
          >
            Next
            <span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
          </button>
        </div>
      </Centered>
    );
  }
}
