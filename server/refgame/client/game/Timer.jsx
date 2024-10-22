import React from "react";

import { StageTimeWrapper } from "meteor/empirica:core";
import Timer from "./Timer.jsx";

class timer extends React.Component {
  render() {
    let { game, remainingSeconds, player, setTimeRemaining, round, whose } = this.props;

    setTimeRemaining(remainingSeconds) // "actual" time remaining (out of 60s)

    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - remainingSeconds;
    // var timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = remainingSeconds - 10;

    if (player.get("role") == 'speaker') {
      timeRemainTurn -= game.get("listenTime") // speaker timer is 15s less than listener
    }

    let minutes = ("0" + Math.floor(timeRemainTurn / 60)).slice(-2);
    let seconds = ("0" + (timeRemainTurn - minutes * 60)).slice(-2);
    if (timeRemainTurn < 0) {
      // speaker timer goes to 00:00; listener still has 15s for selection
      minutes = "00"
      seconds = "00"
    }
    // if (round.get("turnCount") == -1) {
    //   minutes = "00"
    //   seconds = "00"
    // }

    const classes = ["timer", "bp3-card"];
    if (whose == "player") {
      if (timeRemainTurn <= 5) {
        classes.push("lessThan5");
      } else if (timeRemainTurn <= 10) {
        classes.push("lessThan10");
      }
    } else {
      classes.push("grey");
    }


    return (
      <div className={classes.join(" ")}>
        {whose == "player"
          ? <h5 className='bp3-heading'>Timer</h5>
          : <h5 className='bp3-heading' style={{ color: '#999999' }}>Partner's</h5>}
        <span className="seconds">{minutes}:{seconds}</span>
      </div>
    );
  }
}

export default (Timer = StageTimeWrapper(timer));
