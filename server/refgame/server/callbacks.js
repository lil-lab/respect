import Empirica from "meteor/empirica:core";
import { names, avatarNames, nameColors } from './constants.js';
import _ from "lodash";
const { MongoClient } = require("mongodb");

const MONGO_URI = Meteor.settings["galaxy.meteor.com"]["env"]["MONGO_URL"];

function typeOf(obj) {
  return {}.toString.call(obj).split(' ')[1].slice(0, -1).toLowerCase();
}

const updatePlayerIdle = function (game, round, stage, player) {
  player.set("blame", player.get("blame") + 1);
  player.set("warningSound", true);
  round.set("blamingWho", player.get("role"));
  console.log("idle handle updated");

  if (player.get("role") == "speaker") {
    round.set("secUntilSend", -1);
    round.set("sawMessage", true);
    round.set("secUntilSeeMessage", -1);
  }
  round.set("secUntilSubmit", -1);
}

// This only calculates bonus for immature rounds
const calculateBonus = function (game, round, player, totalTime) {
  const bonusList = Array.from(game.get("bonusList"));
  const b = bonusList[0] / 100; // now should be 0.16
  // console.log("total time in minutes: " + totalTime);
  const baseBonus = totalTime * b;
  player.set("bonus", player.get("bonus") + baseBonus);
}

const idleExit = function (game, player) {
  if (!player.get('exited')) {
    player.set('exited', true);

    // let blameRole;
    // if (player.get("blame") < 2) {
    //   player.set("showHIT", true);
    //   blameRole = "your partner";
    //   // blameRole = player.get("role") == "listener" ? "speaker" : "listener";
    // } else {
    //   blameRole = "you";
    //   //  blameRole = player.get("role"); 
    // }

    // change to speaker / listener disconnected
    player.set('exitTimeoutId', player.exit("The game ended because a player stopped participating."));
    game.set("exitFromTimeout", true);
  }
}


// onGameStart is triggered once per game before the game starts, and before
// the first onRoundStart. It receives the game and list of all the players in
// the game.
Empirica.onGameStart((game) => {
  console.log("game is starting")
  const players = game.players;
  console.debug("game ", game._id, " started");

  const teamColor = game.treatment.teamColor;

  players.forEach(player => {
    if (players[0]._id === player._id) {
      player.set('partner', players[1]._id);
    }
    else {
      player.set('partner', players[0]._id);
    }
  });
  players.forEach((player, i) => {
    player.set("name", names[i]);
    player.set("avatar", `/avatars/jdenticon/${avatarNames[teamColor][i]}`);
    player.set("avatarName", avatarNames[teamColor][i]);
    player.set("nameColor", nameColors[teamColor][i]);
    // // player.set("roleIdx", i);
    // const b = Array.from(game.get("bonusList"))[0] / 100;
    // const lobbyTime = (player.get("lobbyEndAt") - player.get("lobbyStartAt")) / 1000; // in seconds
    // player.set("lobbyWaitTime", lobbyTime);
    // player.set("bonus", lobbyTime / 60 * b);
    // player.set("bonus", 0);
    player.set("view", -1);
    player.set("blame", 0);
    player.set("showHIT", false);
    player.set("warningSound", false); // play the warning sound for the idler
    player.set("addPoints", 0);
    // player.set("clicked", 0);
    // player.set("clicks", []);
  });
  console.log("finish onGameStart");
});

// onRoundStart is triggered before each round starts, and before onStageStart.
// It receives the same options as onGameStart, and the round that is starting.
Empirica.onRoundStart((game, round) => {
  const players = game.players;
  // const rooms = game.get('rooms')[round.index];
  round.set("chat", []);
  round.set("keyPressed", false);
  round.set("clicks", []);
  // players.forEach(player => {
  //   if (game.get("botGame")) {
  //     if (typeof player.bot != 'undefined') {
  //       player.set("role", "listener");
  //     } else {
  //       player.set("role", "speaker");
  //     }
  //   } else {
  //     player.set("role", round.get("roles")[player.get("roleIdx")])
  //     // player.set('clicked', 0);
  //     player.set("view", -1); // viewing the history of turn of index view
  //   }
  // });
  round.set("nextRound", false);
});

// onStageStart is triggered before each stage starts.
// It receives the same options as onRoundStart, and the stage that is starting.
Empirica.onStageStart((game, round, stage) => {
  const players = game.players;
  console.debug("Round ", stage.name, "game", game._id, " started");
  stage.set("log", [
    {
      verb: stage.name + "Started",
      roundId: stage.name,
      at: new Date(),
    },
  ]);
  round.set("clicked", false);
  // round.set("blamingWho", "");
  round.set("whoseTurn", 0);
  round.set("submitted", false);
  round.set("sawMessage", false);
  round.set("autoSubmitted", false); // this is for listener autoSubmitted
  game.set("updated", false); // this is for idle player updates
  if (game.get("botGame")) { round.set("botActed", false); }
  console.log("finished onStageStart");
});

// onStageEnd is triggered after each stage.
// It receives the same options as onRoundEnd, and the stage that just ended.
Empirica.onStageEnd((game, round, stage) => {
  // console.log("stage end");
  // console.log(round.get("timeList"))
  // round.set("whoseTurn", 0);
  const players = game.players;
  var timeList = round.get("timeList");
  // console.log(new Date(stage.get("log")[0]["at"]));
  // console.log(new Date());
  // console.log(new Date(stage.get("log")[0]["at"]) - new Date());
  round.set("turnCount", round.get("turnCount") + 1);
  timePassTurn = (new Date() - new Date(stage.get("log")[0]["at"])) / 1000;
  timeList.push(timePassTurn);
  round.set("timeList", timeList);

  // Check if someone idled
  if (Array.from(round.get("chat")).length + round.get("forgiveSpeaker") <= round.get("turnCount") - 1) {
    round.set("forgiveSpeaker", round.get("forgiveSpeaker") + 1);
    players.forEach(player => {
      if (player.get("role") == "speaker") {
        round.set("roundStatus", "speaker");
        updatePlayerIdle(game, round, stage, player);
      }
    });
  } else if (!round.get("clicked")) {
    players.forEach(player => {
      if (player.get("role") == "listener") {
        round.set("roundStatus", "listener");
        updatePlayerIdle(game, round, stage, player);
      }
    });
  }

  var turnDict = {};
  turnDict["secUntilSend"] = round.get("secUntilSend");
  turnDict["secUntilSubmit"] = round.get("secUntilSubmit");
  turnDict["secUntilSeeMessage"] = round.get("secUntilSeeMessage");
  if (round.get("secUntilSend") == -1 || round.get("secUntilSubmit") == -1) {
    turnDict["secBetweenSendAndSubmit"] = -1;
    turnDict["secBetweenSendAndSee"] = -1;
  } else {
    turnDict["secBetweenSendAndSubmit"] = turnDict["secUntilSubmit"] - turnDict["secUntilSend"];
    turnDict["secBetweenSendAndSee"] = Math.max(0, turnDict["secUntilSeeMessage"] - turnDict["secUntilSend"]);
  }

  turnDict["clicks"] = Array.from(round.get("clicks"));
  var turnList = Array.from(round.get("turnList"));
  turnList.push(turnDict);
  round.set("turnList", turnList);


  const onePlayerIdles = players.some(player => player.get("blame") === 1);
  console.log("onePlayerIdles: "+ onePlayerIdles);
  if (onePlayerIdles) {
    players.forEach(player => {
      const timeList = Array.from(round.get("timeList"));
      console.log("timeList: " + timeList);
      const totalTime = timeList.reduce((sum, num) => sum + num, 0) / 60;
      // console.log("total time in minutes: " + totalTime);
      calculateBonus(game, round, player, totalTime);
      idleExit(game, player);
    });
  }
});

// onRoundEnd is triggered after each round.
Empirica.onRoundEnd((game, round) => {
  const players = game.players;
  const bonusList = Array.from(game.get("bonusList"));
  const b = bonusList[0] / 100; // now should be 0.16
  const k1 = bonusList[1] / 100 // for the numerator
  const k2 = bonusList[2]  // for the denominator
  const c = bonusList[3] / 100  // for the y=kx+a
  const d = bonusList[4] / 100; // credit for partial success (7)

  // green: number of correct choices (each target is counted at most once)
  const green = round.get("rightCount");
  // red: If the same thing is selected wrongly twice, it gets counted twice
  const red = round.get("wrongCount");
  // var turnCount = round.get("turnCount");
  // var numRandom = game.get("numRandom");
  const numTarget = Array.from(round.get("target")).length;
  const timeList = Array.from(round.get("timeList"));
  console.log("timeList: " + timeList);
  const totalTime = timeList.reduce((sum, num) => sum + num, 0) / 60;
  // const totalTime = (timeList[0] - timeList[timeList.length - 1]) / 60;
  console.log("total time in minutes: " + totalTime);
  const baseBonus = totalTime * b;
  const successBonus = (green == numTarget && red == 0) ? -k1 / k2 * totalTime + c : 0;
  const partialSuccessBonus = Math.max(0, green-red) * d;
  const totalBonus = baseBonus + successBonus + partialSuccessBonus;

  console.log("baseBonus: " + baseBonus);
  console.log("successBonus: " + successBonus);
  console.log("partialSuccessBonus: " + partialSuccessBonus);
  console.log("totalBonus: " + totalBonus);
  console.log("hourly pay: " + (totalBonus / totalTime * 60));

  if (green == numTarget && red == 0) {
    round.set("roundStatus", "successful");
  } else {
    round.set("roundStatus", "unsuccessful");
  }

  // Update player scores 
  players.forEach(player => {
    const f = player.get("expert") ? 1 : 0.9;
    player.set("bonus", (player.get("bonus") + totalBonus) * f);

    const partner = _.find(
      game.players,
      (p) => p._id === player.get("partner")
    );
    if (green == numTarget && red == 0 && !player.get("expert")) {
      let addPoints;
      if (partner.get("expert") || player.get("role") == "speaker") {
        addPoints = 1;
      } else {
        addPoints = 0.5;
      }
      player.set("addPoints", addPoints);
    }

  });

  // round.set('response', players[0].get('clicked'));
  // round.set('correct', correctAnswer == players[0].get('clicked')["path"]);
});

// onRoundEnd is triggered when the game ends.
// It receives the same options as onGameStart.
Empirica.onGameEnd((game) => {
  console.debug("The game", game._id, "has ended");
});

// ===========================================================================
// => onSet, onAppend and onChanged ==========================================
// ===========================================================================

// onSet, onAppend and onChanged are called on every single update made by all
// players in each game, so they can rapidly become quite expensive and have
// the potential to slow down the app. Use wisely.
//
// It is very useful to be able to react to each update a user makes. Try
// nontheless to limit the amount of computations and database saves (.set)
// done in these callbacks. You can also try to limit the amount of calls to
// set() and append() you make (avoid calling them on a continuous drag of a
// slider for example) and inside these callbacks use the `key` argument at the
// very beginning of the callback to filter out which keys your need to run
// logic against.
//
// If you are not using these callbacks, comment them out so the system does
// not call them for nothing.

// // onSet is called when the experiment code call the .set() method
// // on games, rounds, stages, players, playerRounds or playerStages.
Empirica.onSet(
  (
    game,
    round,
    stage,
    player, // Player who made the change
    target, // Object on which the change was made (eg. player.set() => player)
    targetType, // Type of object on which the change was made (eg. player.set() => "player")
    key, // Key of changed value (e.g. player.set("score", 1) => "score")
    value, // New value
    prevValue // Previous value
  ) => {
    // Compute score after player clicks
    // if ((targetType == "round") && (key == "keyPressed")) {
    //   if (prevValue == false && value == true) {
    //     round.set("secUntilKeyPress", game.get("turnTime") - round.get("lastKeyPressedTime"));
    //   }
    // }

    // recording the information of a turn into a dictionary and push to a list for this round
    //console.log(targetType);
    //console.log(key);
    //console.log(prevValue);
    //console.log(value);


  }
);

// // onAppend is called when the experiment code call the `.append()` method
// // on games, rounds, stages, players, playerRounds or playerStages.
// Empirica.onAppend((
//   game,
//   round,
//   stage,
//   players,
//   player, // Player who made the change
//   target, // Object on which the change was made (eg. player.set() => player)
//   targetType, // Type of object on which the change was made (eg. player.set() => "player")
//   key, // Key of changed value (e.g. player.set("score", 1) => "score")
//   value, // New value
//   prevValue // Previous value
// ) => {
//   // Note: `value` is the single last value (e.g 0.2), while `prevValue` will
//   //       be an array of the previsous valued (e.g. [0.3, 0.4, 0.65]).
// });

// // onChange is called when the experiment code call the `.set()` or the
// // `.append()` method on games, rounds, stages, players, playerRounds or
// // playerStages.
// Empirica.onChange((
//   game,
//   round,
//   stage,
//   players,
//   player, // Player who made the change
//   target, // Object on which the change was made (eg. player.set() => player)
//   targetType, // Type of object on which the change was made (eg. player.set() => "player")
//   key, // Key of changed value (e.g. player.set("score", 1) => "score")
//   value, // New value
//   prevValue, // Previous value
//   isAppend // True if the change was an append, false if it was a set
// ) => {
//   // `onChange` is useful to run server-side logic for any user interaction.
//   // Note the extra isAppend boolean that will allow to differenciate sets and
//   // appends.
// });
