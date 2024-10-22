import React from "react";

import SocialInteractions from "./SocialInteractions.jsx";
import Task from "./Task.jsx";

// const roundSound = new Audio("experiment/round-sound.mp3");
// const gameSound = new Audio("experiment/bell.mp3");
// const gameSound = new Audio("experiment/start-game.mp3");
const crypto = require("crypto");

const cancelTimeout = function (player) {
  const id = player.get("exitTimeoutId");
  if (id) {
    Meteor.clearTimeout(id);
    player.set("exitTimeoutId", null);
  }
};

export default class Round extends React.Component {
  componentDidMount() {
    const { game } = this.props;
    if (game.get("justStarted")) {
      //play the bell sound only once when the game starts
      const gameSound = new Audio("experiment/start-game.mp3");
      gameSound.autoplay = true;
      gameSound.play();
      game.set("justStarted", false);
    } else {
      //roundSound.play();
    }
  }
  // TODO: add intro screan here!!!

  constructor() {
    super();
    this.state = { timeRemaining: -1 };
  }

  getTimeRemaining = (time) => {
    this.setState({ timeRemaining: time });
  };

  render() {
    const { round, stage, player, game } = this.props;
    const allPlayersOnline = game.players.every(
      (player) => player.online || player.bot
    );
    const anyPlayersExited = game.players.some((player) =>
      player.get("exited")
    );
    const partner = _.find(
      game.players,
      (p) => p._id === player.get("partner")
    );

    // Record player IP address in hashed form
    if (!player.get("savedIP")) {
      player.set("savedIP", true);
      fetch("https://api.ipify.org?format=json")
        .then((response) => response.json())
        .then((data) => {
          player.set(
            "hashedIP",
            crypto.createHash("md5").update(data.ip).digest("hex")
          );
          // console.log(data.ip);
          // console.log("hashedIP: " + player.get("hashedIP"));
        })
        .catch((error) => {
          console.log("Game start");
        });
    }

    if (!player.get("exited")) {
      if (player.get("warningSound")) {
        const warningSound = new Audio("experiment/bell.mp3");
        warningSound.autoplay = true;
        warningSound.play();
        player.set("warningSound", false);
        console.log("played warning sound");
      }

      // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - this.state.timeRemaining;
      // const timeRemainTurn = game.get("turnTime") - timePassTurn;
      var timeRemainTurn = this.state.timeRemaining - 10;

      // speaker doesn't send message
      if (
        timeRemainTurn == game.get("listenTime") - 1 &&
        player.get("role") == "listener"
      ) {
        const speakerMsgs = _.filter(round.get("chat"), (msg) => {
          return msg.role == "speaker";
        });
        if (
          speakerMsgs.length + round.get("forgiveSpeaker") <=
          round.get("turnCount")
        ) {
          if (!game.get("updated")) {
            game.set("updated", true);
            console.log("exit because speaker didn't send any messages");
            Meteor.setTimeout(() => partner.stage.submit(), 500);
            Meteor.setTimeout(() => player.stage.submit(), 500);
          }
        }
      }

      // listener doesn't submit
      else if (
        player.get("role") == "speaker" &&
        timeRemainTurn > -2 &&
        timeRemainTurn < 0 &&
        !round.get("clicked")
      ) {
        if (!game.get("updated")) {
          game.set("updated", true);
          console.log("exit because listener didn't submit");
          Meteor.setTimeout(() => partner.stage.submit(), 500);
          Meteor.setTimeout(() => player.stage.submit(), 500);
        }
      }
    }
    return (
      <div className="round">
        <SocialInteractions
          game={game}
          round={round}
          stage={stage}
          player={player}
          getTimeRemaining={this.getTimeRemaining}
          timeRemaining={this.state.timeRemaining}
        />
        <Task
          game={game}
          round={round}
          stage={stage}
          player={player}
          timeRemaining={this.state.timeRemaining - 10}
        />
      </div>
    );
  }
}
