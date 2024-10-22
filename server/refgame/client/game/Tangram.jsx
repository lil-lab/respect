import React from "react";

function htmlToElements(html) {
  var template = document.createElement("template");
  template.innerHTML = html;
  return template.content.childNodes;
}

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

export default class Tangram extends React.Component {
  handleClick = (e) => {
    const { game, tangram, tangram_num, stage, player, round, timeRemaining } =
      this.props;
    const speakerMsgs = _.filter(round.get("chat"), (msg) => {
      return (msg.role == "speaker") & (msg.playerId == player.get("partner"));
    });
    const partner = _.find(
      game.players,
      (p) => p._id === player.get("partner")
    );

    // only register click for listener and only after the speaker has sent a message
    if (
      (speakerMsgs.length > 0) &&
      (player.get("role") == "listener") &&
      (round.get("whoseTurn") == 1) // timeRemaining < listenTime
    ) {
      var clicks = Array.from(round.get("clicks"));
      var target = Array.from(round.get("target"));
      round.set("clicked", true);
      const tangram_path = tangram["path"];
      if (clicks.includes(tangram_path)) {
        // round.set("clicked", round.get("clicked") - 1);
        const index = clicks.indexOf(tangram_path);
        clicks.splice(index, 1);
        // var updatedList = clicks.filter(item => item !== tangram);
        partner.set("clicks", clicks);
        // console.log("partner has clicked", partner.get("clicks"));
        round.set("clicks", clicks);
      }
      else {
        // round.set("clicked", round.get("clicked") + 1);
        clicks.push(tangram_path);
        partner.set("clicks", clicks);
        // console.log("partner has clicked", partner.get("clicks"));
        round.set("clicks", clicks);
      }
    }
  };

  render() {
    const {
      game,
      tangram,
      tangram_num,
      round,
      stage,
      player,
      target, // now a list (array) of paths
      timeRemaining,
      tangramsPerRow,
      contextSize
    } = this.props;
    const tangram_path = tangram["path"];
    // { console.log("tangram path", tangram_path) }
    const tangram_data = tangram["data"];
    var colors = undefined;
    if ("coloring-reassigment" in tangram) {
      colors = tangram["coloring-reassigment"];
    }
    const mystyle = {
      backgroundSize: "cover",
      width: "auto",
      height: "auto",
      display: "inline-block",
      margin: "15px",
      // gridRow: currentPosition["row"],
      // gridColumn: currentPosition["column"],
    };

    // Highlight target object for speaker at selection stage
    if (
      (target.includes(tangram_path)) & (player.get("role") == "speaker")
    ) {
      _.extend(mystyle, {
        outline: "10px solid #000",
        outlineOffset: "4px",
        zIndex: "9",
      });
    }

    // Highlight clicked object in green if correct; red if incorrect

    // set it to round.get("clicks") if view=-1, else, set it to the clicks at that turn
    const view = player.get("view");
    const turnList = round.get("turnList");
    if (view == -1 || typeof turnList[view] === 'undefined') {
      var clicks = Array.from(round.get("clicks")); // now it's the paths of clicks
    } else {
      // console.log(turnList[view]);
      var clicks = Array.from(turnList[view]["clicks"]); // the history from the Index-th turn
    }
    // var clicks = Array.from(round.get("clicks")); // now it's the paths of clicks


    if (
      (clicks.includes(tangram_path)) && (player.get("role") == "listener")
    ) {
      _.extend(mystyle, {
        outline: "10px solid #000",
        outlineOffset: "4px",
        zIndex: "9",
      });
    } else if ((round.get("whoseTurn") == 0 || view > -1) && round.get("turnCount") > 0 && (player.get("role") == "speaker")) {
      var correct = target.includes(tangram_path); // if this one is correct
      const color = correct ? "green" : "red";
      if (clicks.includes(tangram_path)) { // if this one is clicked
        _.extend(mystyle, {
          outline: `10px solid ${color}`,
          outlineOffset: "4px",
          zIndex: "9",
        });
      }
    }

    // This is when it's correct and we will move on to the next round
    if (round.get("submitted") && compareArrays(target, clicks) && (clicks.includes(tangram_path))) {
      _.extend(mystyle, {
        outline: "10px solid green",
        outlineOffset: "4px",
        zIndex: "9",
      })
    }

    var elements = htmlToElements(tangram_data);
    for (let i = 0; i < elements.length; i++) {
      if (elements[i].nodeName == "svg") {
        var svg = elements[i];
      }
    }
    var childrenArray = Array.prototype.slice.call(svg.childNodes);

    var bodyElement = document.evaluate(
      "/html/body",
      document,
      null,
      XPathResult.FIRST_ORDERED_NODE_TYPE,
      null
    ).singleNodeValue;

    var numRows = Math.ceil(contextSize / tangramsPerRow);
    var minSize, tangramWidth, tangramHeight;
    tangramWidth = bodyElement.offsetWidth / 2 / (tangramsPerRow + 0.5);
    tangramHeight = bodyElement.offsetHeight / 2 / (numRows + 1);
    minSize = Math.min(tangramWidth, tangramHeight);
    tangramWidth = minSize;
    tangramHeight = minSize;

    return (
      <div id={tangram_path} onClick={this.handleClick} style={mystyle}>
        <svg
          baseProfile="full"
          viewBox={svg.getAttribute("viewBox")}
          width={tangramWidth}
          height={tangramHeight}
          xmlns="http://www.w3.org/2000/svg"
        >
          {childrenArray.map((node, index) => {
            if (node.nodeName == "polygon") {
              if (
                colors === undefined ||
                !(node.getAttribute("id") in colors)
              ) {
                var colorFill = "#1C1C1C"; //node.getAttribute("fill"), "black", lightgray
              } else {
                var colorFill = colors[node.getAttribute("id")];
              }
              var id = tangram_path + "_" + node.getAttribute("id");
              return (
                <polygon
                  key={id}
                  id={id}
                  fill={colorFill}
                  points={node.getAttribute("points")}
                  stroke={colorFill} //{node.getAttribute("stroke")}
                  strokeWidth={"2"} //{node.getAttribute("strokeWidth")}
                  transform={node.getAttribute("transform")}
                />
              );
            }
          })}
        </svg>
      </div>
    );
  }
}
