import React, { useEffect } from "react";
import Author from "./Author";
import Timer from "./Timer";

export default class ChatLog extends React.Component {
  state = { comment: "" };

  checkTimeRemaining = () => {
    const { game, round, timeRemaining } = this.props;
    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // const timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = timeRemaining;
    if (timeRemainTurn === game.get("listenTime")) {
      // console.log("speaker auto sent");
      this.handleSubmit();
    }
  }

  handleEmoji = (e) => {
    e.preventDefault();
    const text = e.currentTarget.value;
    console.log(text);
    const { game, round, player, stage, timeRemaining } = this.props;
    //const room = player.get('roomId')
    console.log(stage);
    console.log(timeRemaining);
    round.append("chat", {
      text,
      playerId: player._id,
      target: round.get("target"),
      role: player.get("role"),
      type: "message",
      time: new Date(),
      secUntilSend: game.get("turnTime") - timeRemaining,
    });
  };

  handleChange = (e) => {
    const el = e.currentTarget;
    this.setState({ [el.name]: el.value });

    const { round, timeRemaining } = this.props;
    // round.set("lastKeyPressedTime", timeRemaining);
    // round.set("keyPressed", true);
  };

  handleSubmit = (e) => {
    if (e) e.preventDefault();
    const text = this.state.comment.trim();
    const { game, round, player, stage, timeRemaining } = this.props;

    if (text !== "") {
      console.log("message sent after seconds: ", game.get("turnTime") - timeRemaining);
      round.set("whoseTurn", 1);
      round.append("chat", {
        text,
        playerId: player._id,
        // roomId: room,
        target: round.get("target"),
        role: player.get("role"),
        time: new Date(),
        secUntilSend: game.get("turnTime") - timeRemaining,
      });
      this.setState({ comment: "" });
      round.set("secUntilSend", game.get("turnTime") - timeRemaining)
    }
  };

  // useEffect(()=>{
  //   const { game, round, player, stage, timeRemaining } = this.props;
  //   if(timeRemaining === 0) {
  //     handleSubmit();
  //   }
  // }, [timeRemaining]);

  render() {
    const { comment } = this.state;
    const { game, messages, player, round, stage, timeRemaining } = this.props;

    var placeholder = "Enter chat message";
    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // var timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = timeRemaining;
    var disableAttribute = null;

    if (player.get("role") == "speaker") {
      // if (round.get("chatMode") == "single-utterance-unidirectional") {
      //   if (messages.length == 0) {
      //     placeholder = "You can send only one message";
      //   } else {
      //     disableAttribute = "disabled";
      //     placeholder = "You have already sent one message";
      //   }
      // }
      if (round.get("chatMode") == "multi-utterance-unidirectional") {
        const bool1 = (round.get("whoseTurn") == 0);
        const bool2 = (round.get("turnCount") < game.get("numTurns"));
        // console.log(bool1, bool2);
        if (bool1 && bool2) {
          if (round.get("rightCount") == Array.from(round.get("target")).length && round.get("wrongCount") == 0) {
            disableAttribute = "disabled";
            placeholder = "You've succeeded this round.";
          } else {
            placeholder = "You can send only one message each turn.";
          }

        } else {
          disableAttribute = "disabled";
          placeholder = "You have already sent one message this turn.";
        }
      }
    }


    if (player.get("role") == "listener") {
      if (
        round.get("chatMode") == "multi-utterance-unidirectional" ||
        round.get("chatMode") == "single-utterance-unidirectional"
      ) {
        disableAttribute = "disabled";
        placeholder = "You are the listener. You can't send a message";
      }

      if (messages.length + round.get("forgiveSpeaker") > round.get("turnCount") && !round.get("sawMessage")) {
        // console.log(messages.length, round.get("forgiveSpeaker"), round.get("turnCount"));
        round.set("sawMessage", true);
        round.set("secUntilSeeMessage", game.get("turnTime") - timeRemaining);
        console.log("Listener saw speaker message after seconds: ", round.get("secUntilSeeMessage"));
      }
    }

    if (
      timeRemainTurn !== -1 &&
      timeRemainTurn <= game.get("listenTime") &&
      player.get("role") === "speaker"
    ) {
      disableAttribute = "disabled";
      placeholder =
        "The next " + game.get("listenTime") + "s is the selection stage. You can't send messages.";
    }

    if (
      round.get("chatMode") != "multi-utterance-backchannel" ||
      player.get("role") == "speaker"
    ) {
      if (Array.from(round.get("chat")).length + round.get("forgiveSpeaker") <= round.get("turnCount")) { this.checkTimeRemaining(); }
      return (
        <div className="chat bp3-card">
          <Messages messages={messages} player={player} />
          <form onSubmit={this.handleSubmit}>
            <div className="bp3-control-group">
              <input
                name="comment"
                type="text"
                className="bp3-input bp3-fill"
                placeholder={placeholder}
                value={comment}
                onChange={this.handleChange}
                autoComplete="off"
                disabled={disableAttribute}
              />
              <button
                type="submit"
                className="bp3-button bp3-intent-primary"
                disabled={disableAttribute}
              >
                Send
              </button>
            </div>
          </form>
        </div>
      );
    } else {
      return (
        <div className="chat bp3-card">
          <Messages messages={messages} player={player} />
          <div className="bp3-button-group bp3-fill bp3-fill">
            <button
              type="button"
              className="bp3-button"
              value="&#10060;"
              onClick={this.handleEmoji}
            >
              &#10060;
            </button>
            <button
              type="button"
              className="bp3-button"
              value="&#129300;"
              onClick={this.handleEmoji}
            >
              &#129300;
            </button>
            <button
              type="button"
              className="bp3-button"
              value="	&#9989;
              "
              onClick={this.handleEmoji}
            >
              &#9989;
            </button>
            <button
              type="button"
              className="bp3-button"
              value="	&#128514;"
              onClick={this.handleEmoji}
            >
              &#128514;
            </button>
          </div>
        </div>
      );
    }
  }
}

const chatSound = new Audio("experiment/unsure.mp3");
class Messages extends React.Component {
  componentDidMount() {
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  componentDidUpdate(prevProps) {
    if (prevProps.messages.length < this.props.messages.length) {
      this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
      //chatSound.play();
    }
  }
  render() {
    const { messages, player } = this.props;



    return (
      <div className="messages" ref={(el) => (this.messagesEl = el)}>
        {messages.length === 0 ? (
          <div className="empty">No messages yet...</div>
        ) : null}
        {messages.map((message, i) => (
          <Message
            key={i}
            message={message}
            self={message.subject ? player._id === message.subject._id : null}
            shouldBold={i == player.get("view")}
          />
        ))}
      </div>
    );
  }
}

class Message extends React.Component {
  render() {
    const { text, subject } = this.props.message;
    const { self, shouldBold } = this.props;
    return (
      // <div className="message">
      //   <Author player={subject} self={self} />
      //   {text}
      // </div>

      // <div className={`message ${showBox ? 'circle-box' : ''}`}>
      //   <Author player={subject} self={self} />
      //   {text}
      // </div>

      <div className={`message ${shouldBold ? 'bold' : ''}`}>
        <Author player={subject} self={self} />
        {text}
      </div>
    );
  }
}
