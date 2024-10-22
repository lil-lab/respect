import React from "react";

import Tangram from "./Tangram.jsx";
import { Centered } from "meteor/empirica:core";
// import Timer from "./Timer.jsx";
// import { HTMLTable } from "@blueprintjs/core";
// import { StageTimeWrapper } from "meteor/empirica:core";

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

export default class Task extends React.Component {
  constructor(props) {
    super(props);

    // We want each participant to see tangrams in a random but stable order
    // so we shuffle at the beginning and save in state
    this.state = {
      activeButton: false,
    };
  }

  checkTimeRemaining = () => {
    const { game, round, timeRemaining } = this.props;
    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // const timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = timeRemaining;
    if (timeRemainTurn === 0 && round.get("clicked")) {
      round.set("autoSubmitted", true);
      console.log("listener auto submitted");
      this.submitClick();
    }
  }


  submitClick = (e) => {
    const { game, tangram, tangram_num, stage, player, round, timeRemaining } =
      this.props;
    const partner = _.find(
      game.players,
      (p) => p._id === player.get("partner")
    );


    // only allow submit if this is the listener's turn
    var clicks = Array.from(round.get("clicks"));
    var target = Array.from(round.get("target"));
    //round.get("wrongCount") + 

    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // const timeRemainTurn = game.get("turnTime") - timePassTurn;
    const timePassTurn = game.get("turnTime") - timeRemaining;
    var timeRemainTurn = timeRemaining;
    round.set("secUntilSubmit", timePassTurn);
    // Handle the "submit" click (moved to on stage end)
    round.set("clicked", true);
    round.set("submitted", true);
    // round.set("turnCount", round.get("turnCount") + 1);
    round.set("blamingWho", "");

    // var timeList = round.get("timeList");
    // timeList.push(timePassTurn);
    // round.set("timeList", timeList);

    console.log("submitted after seconds: ", timePassTurn);

    // Go to next round if choose correctly or has reached maximum number of turns
    if (compareArrays(target, clicks) || round.get("turnCount") == game.get("numTurns") - 1) {
      round.set("wrongCount", clicks.filter(item => !target.includes(item)).length);
      round.set("rightCount", clicks.filter(item => target.includes(item)).length);
      // round.set("turnCount", -1);
      round.set("clickedTime", new Date());
      console.log("exit because listener submitted");
      player.set("warningSound", false);
      partner.set("warningSound", false);
      game.set("consecutiveIdle", 0);
      player.set("showHIT", true);
      partner.set("showHIT", true);
      round.set("nextRound", true);
      Meteor.setTimeout(() => player.stage.submit(), 3000); //, 3000
      Meteor.setTimeout(() => partner.stage.submit(), 3000); //, 3000

    } else {
      Meteor.setTimeout(() => player.stage.submit(), 200);
      Meteor.setTimeout(() => partner.stage.submit(), 200);
    }
  };


  render() {
    const { game, round, stage, player, timeRemaining } = this.props;
    const target = round.get("target"); // Now a list of paths

    const contextSize = round.get("numTangrams");
    const tangramsPerRow = Math.ceil(Math.sqrt(contextSize));

    // const timePassTurn = Array.from(round.get("timeList"))[round.get("turnCount")] - timeRemaining;
    // const timeRemainTurn = game.get("turnTime") - timePassTurn;
    var timeRemainTurn = timeRemaining;
    if (player.get("role") == "speaker") {
      var tangramURLs = round.get("tangrams")[0];
    } else {
      var tangramURLs = round.get("tangrams")[1];
    }
    if (player.get("role") == "listener" && !round.get("autoSubmitted")) {
      this.checkTimeRemaining();
    }// only aubmit if not autoSubmitted yet

    // var buttons = [];
    // if (round.get('turnCount') == -1) {
    //   buttons = Array.from({ length: 0 }, (_, index) => `Turn ${index + 1}`);
    // } else {
    //   buttons = Array.from({ length: round.get("turnCount") }, (_, index) => `Turn ${index + 1}`);
    // }
    // const buttonElements = buttons.map((button, index) => (
    //   <button
    //     key={index}
    //     onMouseEnter={() => player.set("view", index)}
    //     onMouseLeave={() => player.set("view", -1)}
    //     style={{
    //       background: 'darkgray',
    //       color: 'white',
    //       margin: '3px',
    //       padding: '4px 6px',
    //       border: 'none',
    //       borderRadius: '3px',
    //     }}
    //   >
    //     {button}
    //   </button>
    // ));

    // console.log(tangramURLs);
    let tangramsToRender;
    if (tangramURLs) {
      tangramsToRender = tangramURLs.map((tangram, i) => (
        <Tangram
          key={tangram["path"]}
          tangram={tangram}
          tangram_num={i}
          round={round}
          stage={stage}
          game={game}
          player={player}
          target={target} // now a set
          timeRemaining={timeRemaining}
          tangramsPerRow={tangramsPerRow}
          contextSize={contextSize}
        />
      ));
    }


    return (
      <div className="task" style={{ display: "inline-block" }}>
        <div className="board">
          <h1 className="roleIndicator">
            {" "}
            You are the{" "}
            <span
              style={{
                color: player.get("role") === "speaker" ? "IndianRed" : "SteelBlue",
                fontWeight: "bold",
              }}
            >
              {player.get("role")}
            </span>
            .
          </h1>


          <div className="all-tangrams">
            <div className="tangrams">
              {Array.from({ length: Math.ceil(contextSize / tangramsPerRow) }, (row, rowIndex) => (
                <div style={{ display: "flex", justifyContent: "center" }}>
                  {tangramsToRender.slice(rowIndex * tangramsPerRow, Math.min((rowIndex * tangramsPerRow) + tangramsPerRow), contextSize).map((tangram, tangramIndex) => (
                    <div>
                      {tangram}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* <div className="all-tangrams">
            <div className="tangrams" style={{ display: "flex", flexWrap: "wrap", justifyContent: "center" }}>
              {tangramsToRender.map((tangram, tangramIndex) => (
                <div style={{ margin: "0 10px", flex: "1 0 200px", maxWidth: "200px", maxHeight: "200px" }}>
                  {tangram}
                </div>
              ))}
            </div>
          </div> */}

          {(player.get("role") == "listener") && (<div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
            <button
              ref={this.myButton}
              onClick={this.submitClick}
              style={{
                backgroundColor: 'rgb(30, 100, 140)',
                color: 'white',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '5px',
                opacity: (round.get("whoseTurn") === 0 || round.get("submitted")) ? 0.5 : 1,
                pointerEvents: (round.get("whoseTurn") === 0 || round.get("submitted")) ? 'none' : 'auto',
              }}
              disabled={(round.get("whoseTurn") === 0 || round.get("submitted"))}
            >
              Submit
            </button>

          </div>)}

          {/* This is the part about viewing history */}
          {/* {(round.get("turnCount") > 0) && <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '20px' }}>
            <p style={{ fontWeight: 'bold', }}>Hover to see turn history:</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
              {buttonElements}
            </div>
          </div>} */}
        </div>
      </div >
    );
  }
}
