import React from "react";
import EventLog from "./EventLog";
import ChatLog from "./ChatLog";
import Timer from "./Timer";

function compareArrays(array1, array2) {
  if (array1.length !== array2.length) {
    return false;
  }

  for (let i = 0; i < array1.length; i++) {
    if (!(array2.includes(array1[i]))) { return false; }
    if (!(array1.includes(array2[i]))) { return false; }
  }
  return true;
}

export default class SocialInteractions extends React.Component {
  renderPlayer(player, self = false) {
    var key, color, avatar, playerName;
    if (!player) {
      key = undefined;
      color = undefined;
      avatar = undefined;
      playerName = undefined;
    }
    else {
      key = player._id;
      color = player.get("nameColor");
      avatar = player.get("avatar");
      playerName = player.get("name");
    }
    return (
      <div className="player" key={key}>
        <span className="image"></span>
        <img src={avatar} />
        <span className="name" style={{ color: color }}>
          {playerName}
          {self ? " (You)" : " (Partner)"}
        </span>
      </div>
    );
  }

  constructor() {
    super();
    this.state = { timeRemaining: -10 };
  }

  setTimeRemaining = (time) => {
    this.props.getTimeRemaining(time); // pass time remaining up to Round component
    this.setState({ timeRemaining: time });
  }



  render() {
    const { game, round, stage, player, timeRemaining } = this.props;
    const partnerId = player.get('partner');
    const partner = _.filter(game.players, p => p._id === partnerId)[0];
    const messages = round.get("chat")
      .filter(({ playerId }) => playerId === partnerId || playerId === player._id)
      .map(({ text, playerId }) => ({
        text,
        subject: game.players.find(p => p._id === playerId)
      }));
    const events = stage.get("log").map(({ subjectId, ...rest }) => ({
      subject: subjectId && game.players.find(p => p._id === subjectId),
      ...rest
    }));
    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // const timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = timeRemaining - 10;
    var target = Array.from(round.get("target"));

    const speakerMsgs = _.filter(round.get("chat"), (msg) => {
      return msg.role == "speaker";
    });

    let feedback = "";
    const blamingWho = round.get("blamingWho");
    const whoseTurn = round.get("whoseTurn");

    if (round.get("submitted")) {
      if (compareArrays(target, round.get("clicks"))) {
        // If they got things correctly
        feedback = "Correct! You earned the correctness bonus!";
      }
    }

    if (whoseTurn == 0) {

      // displaying feedback when someone idles in the previous turn
      if (blamingWho != "") {
        var feedbackWarning = false;
        //round.get("turnCount") > 0 && round.get("turnCount") < game.get("numTurns")
        // Displaying the blame
        if (blamingWho == "speaker") {
          if (player.get("blame") == 2 || partner.get("blame") == 2) {
            feedback = "The speaker didn't give a description again. This game will be cancelled.";
          }
          else if (player.get("role") == "speaker") {
            feedback = "You didn't type anything last turn. If you do not type anything again, you will not receive pay. Please enter a message.";
            feedbackWarning = true;
          } else {
            feedback = "Sorry, the speaker didn't give a description. They will have one more chance before the game is cancelled.";
          }
        }
        else if (blamingWho == "listener") {
          if (player.get("blame") == 2 || partner.get("blame") == 2) {
            feedback = "The listener didn't make a selection again. This game will be cancelled.";
          }
          else if (player.get("role") == "speaker") {
            feedback = "Sorry, the listener didn't make a selection. They will have one more chance before the game is cancelled. Please enter a message.";
          } else {
            feedback = "You didn't select anything last turn. If you do not select anything again, you will not receive pay. Please make your guesses after the speaker’s message.";
            feedbackWarning = true;
          }
        }
      }
      // Normal choosing but incorrect selection
      else if (speakerMsgs.length != 0 && !round.get("clicked")) {
        if (player.get("role") == "listener") {
          feedback = "You didn't choose all the targets correctly (Maybe the speaker hasn't described every target yet). Try again in the next turn.";
        } else {
          feedback = "The listener didn't choose all the targets correctly. The green ones are correct, and the red ones are wrong. Please enter a message."
        }
      }
      // Final turn didn't get things correct
      else if (round.get("turnCount") == game.get("numTurns") - 1) {
        feedback = "You didn't get all the targets correctly. You earned the base bonus.";
      }
    }



    return (
      <div className="social-interactions" style={{ width: "30%", display: "inline-block" }}>
        <div className="status">
          <div className="players bp3-card">
            {this.renderPlayer(player, true)}
            {this.renderPlayer(partner)}
          </div>

          <Timer game={game} stage={stage} player={player} setTimeRemaining={this.setTimeRemaining} round={round} whose={"player"} />

          <Timer game={game} stage={stage} player={partner} setTimeRemaining={this.setTimeRemaining} round={round} whose={"partner"} />

          {/* <div className="total-score bp3-card">
            <h5 className='bp3-heading'>Bonus</h5>

            <h2 className='bp3-heading'>${(player.get("bonus") || 0).toFixed(2)}</h2>
          </div> */}
        </div>
        <ChatLog game={game} messages={messages} round={round} stage={stage} player={player} timeRemaining={this.state.timeRemaining - 10} />
        {/* <EventLog events={events} round={round} game={game} stage={stage} player={player} /> */}
        <h3 className={`feedbackIndicator ${(feedback == "Correct! You earned the correctness bonus!") ? 'correct' : ((feedbackWarning) ? 'warning' : '')}`}>
          {feedback}
        </h3>

      </div>
    );
  }
}
