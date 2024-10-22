import Empirica from "meteor/empirica:core";
import { HTTP } from "meteor/http"


const BOT_URI = Meteor.settings["botUri"];

function sendMessage(round, bot, message) {
  const room = bot.get('roomId')
  round.append("chat", {
    text: message, // "I am a bot sending a message"
    playerId: bot._id,
    target: round.get('target')[room],
    role: bot.get('role')
  });
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

function chooseTarget(target_path, game, round, stage, bot) {
  const partner = _.find(game.players, p => p._id === bot.get('partner'));

  // setting stuff for the turn
  console.log(target_path);
  const clicks = Array.from(target_path);
  console.log(clicks);
  round.set("clicks", clicks);
  round.set("clicked", true);
  round.set("submitted", true);
  round.set("blamingWho", "");
  timePassTurn = (new Date() - new Date(stage.get("log")[0]["at"])) / 1000;
  round.set("secUntilSubmit", timePassTurn);
  console.log("submitted after seconds: ", timePassTurn);

  // see if we need to end early and set things for showing stuff
  const target = Array.from(round.get("target"));
  if (compareArrays(target, clicks) || round.get("turnCount") == game.get("numTurns") - 1) {
    round.set("wrongCount", clicks.filter(item => !target.includes(item)).length);
    round.set("rightCount", clicks.filter(item => target.includes(item)).length);
    // round.set("turnCount", -1);
    round.set("clickedTime", new Date());
    console.log("exit because bot submitted");
    partner.set("warningSound", false);
    game.set("consecutiveIdle", 0);
    partner.set("showHIT", true);
    round.set("nextRound", true);
    Meteor.setTimeout(() => bot.stage.submit(), 3000);
    Meteor.setTimeout(() => partner.stage.submit(), 3000);

  } else {
    console.log("Here once!");
    Meteor.setTimeout(() => bot.stage.submit(), 200);
    Meteor.setTimeout(() => partner.stage.submit(), 200);
  }
}


function predictTarget(data, game, round, stage, bot) {
  HTTP.call('POST', BOT_URI, {
    data: data
  }, function (error, response) {
    if (error) {
      console.log('error getting stims from predict target');
      console.log(error);
    } else {
      console.log('got predict_target');
      console.log(response);
      const target_path = JSON.parse(response.content).path;
      chooseTarget(target_path, game, round, stage, bot)
    }
  });
}

function randomChoose(data, game, round, stage, bot) {
  randomNum = Math.floor(Math.random() * 5 + 3);
  randomShuffled = data['image_paths'].sort(() => 0.5 - Math.random());
  randomPaths = (randomShuffled.slice(0, randomNum)).map(item => item.path);
  console.log(randomPaths);
  chooseTarget(randomPaths, game, round, stage, bot);

}

Empirica.bot("bob", {
  // Called during each stage at tick interval (~1s at the moment)
  onStageTick(bot, game, round, stage, secondsRemaining) {

    const timeRemaining = secondsRemaining - 10;

    // if the bot is a listener, predict the target
    if (bot.get('role') === "listener") {
      const speakerMsgs = _.filter(round.get("chat"), msg => {
        return msg.role == 'speaker' & msg.playerId == bot.get('partner')
      })

      if ((speakerMsgs.length + round.get("forgiveSpeaker") > round.get("turnCount"))
        && (timeRemaining % 5 === 3)
        && (round.get("secUntilSend") + timeRemaining < game.get("turnTime") - 2)
        && (!round.get("botActed"))) {

        round.set("botActed", true);
        console.log(speakerMsgs[speakerMsgs.length - 1]);
        const lastMsg = speakerMsgs[speakerMsgs.length - 1].text;
        const allMsg = speakerMsgs.map((msg) => msg.text); //.join(" ");
        const tangramURLs = round.get('tangrams')[1];
        const tangramURLsTrimmed = tangramURLs.map((item) => ({"path":item.path}));
        const target = round.get('target')
        const bot_treatment = game.get('botTreatment');
        const previousSelected = round.get("turnList").map(turn => turn.clicks);

        const data = {
          'image_paths': tangramURLsTrimmed,
          'last_msg': lastMsg,
          'all_msg': allMsg,
          'target': target,
          'bot_treatment': bot_treatment,
          'previous_selected': previousSelected,
          'game_id': game._id, // assume single round game
          'turn_id': round.get("turnCount"),
        }
        console.log(data);
        predictTarget(data, game, round, stage, bot)
        // randomChoose(data, game, round, stage, bot);
      }

      else if ((speakerMsgs.length + round.get("forgiveSpeaker") <= round.get("turnCount")) && timeRemaining == game.get("listenTime") - 1) {
        if (!game.get("updated")) {
          game.set("updated", true);
          const partner = _.find(game.players, p => p._id === bot.get('partner'));
          console.log("exit because speaker didn't send any messages");
          Meteor.setTimeout(() => partner.stage.submit(), 500);
          Meteor.setTimeout(() => bot.stage.submit(), 500);
        }
      }
    }

  }
});
