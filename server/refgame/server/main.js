import Empirica from "meteor/empirica:core";
import "./callbacks.js";
import "./bots.js";
import { tangramsRelativeDir } from "./private.json";
import _ from "lodash";
import path from "path";
import { Mutex } from 'async-mutex';
const { MongoClient } = require("mongodb");

const fs = require("fs"); // read files
const mutex = new Mutex();
const MONGO_URI = Meteor.settings["galaxy.meteor.com"]["env"]["MONGO_URL"];
const CONTROL_TREATMENT_LEN = Number(Meteor.settings["controlTreatmentLen"]);
const CONTROL_TREATMENT_QUEUE = Meteor.settings["controlTreatmentQueue"];
const BASE_CONFIGURATION_PATH = "/games/";
const NUMBER_OF_JSONS = 19;
// var game_index = 0;

// global variables for generating balanced controlled game config
const queue = [];
const seens = {};
const treatment_to_queue = {};


function shuffle(a) {
	var j, x, i;
	var d = _.cloneDeep(a);
	for (i = d.length - 1; i > 0; i--) {
		j = Math.floor(Math.random() * (i + 1));
		x = d[i];
		d[i] = d[j];
		d[j] = x;
	}
	return d;
}

function getOrderedImages(images, tangramsOreder) {
	var new_images = Array(images.length);
	for (i = 0; i < images.length; i++) {
		new_images[tangramsOreder[i]] = images[i];
	}
	return new_images;
}

function controlledGameConfigIndexPerBotTreatment(botTreatment, numConfigs) {
    // Initialize pointer to the beginning of the queue
	if (!(botTreatment in seens)) {
        seens[botTreatment] = 0;
		console.log(botTreatment, "was never seen before. set idex to 0");
    }
    // populate queue with either random or controlled index
	if (queue.length === 0){
		if (CONTROL_TREATMENT_LEN === 0) {
			queue.push(...CONTROL_TREATMENT_QUEUE);
		} else {
			for (let i = 0; i < CONTROL_TREATMENT_LEN; i++) {
				queue.push(Math.floor(Math.random() * numConfigs));
			}
		}
		console.log("queue was never seen before, set queue to ", queue);
	}

	// Initialize treatment queue if not seen before
	if (!(botTreatment in treatment_to_queue)) {
		treatment_to_queue[botTreatment] = shuffle(queue);
		console.log(botTreatment, " was never seen before, set up shuffled queue");
		console.log(treatment_to_queue);
	}

	// Get the index
	let seenIndex = seens[botTreatment]++;
	if (seenIndex === queue.length) {
		seenIndex = 0;
		seens[botTreatment] = 1;
		console.log("reset queue for treatment ", botTreatment);
		console.warn("consider increasing CONTROL_TREATMENT_LEN or restarting server");
	}

	const index = treatment_to_queue[botTreatment][seenIndex];
    console.log("index", index, "for treatment", botTreatment);
    return index;
}

// gameInit is where the structure of a game is defined.  Just before
// every game starts, once all the players needed are ready, this
// function is called with the treatment and the list of players.  You
// must then add rounds and stages to the game, depending on the
// treatment and the players. You can also get/set initial values on
// your game, players, rounds and stages (with get/set methods), that
// will be able to use later in the game.
Empirica.gameInit((game, treatment) => {
	console.log(
		"Game with a treatment: ",
		treatment,
		" will start, with workers",
		_.map(game.players, "id")
	);

	function typeOf(obj) {
		return {}.toString.call(obj).split(" ")[1].slice(0, -1).toLowerCase();
	}

	// Sample whether on the blue team or red team
	game.set("teamColor", treatment.teamColor);

	// Sample whether to use tangram set A or set B
	game.set("gameSet", treatment.gameSet);
	game.set("team", game.players.length > 1);

	// Define the bot treatment for the game
	console.log(treatment.botsCount);
	let botGame = (treatment.botsCount == 1);
	game.set("botGame", botGame);
	let botIdx;
	if (botGame) {
		for (let i = 0; i < 2; i++) {
			if (typeof game.players[i].bot != 'undefined') {
				botIdx = i;
			}
		}
	}



	// I use this to play the sound on the UI when the game starts
	game.set("justStarted", true);

	// Sample the game configuration at random
	basePath = path.join(__meteor_bootstrap__.serverDir, "../web.browser/app"); // directory for folders in /public after built
	var sampledIndex = -1
	if (treatment.controlledGamePerBotTreatment) {
		sampledIndex = controlledGameConfigIndexPerBotTreatment(treatment.botTreatment, treatment.numConfigs);
	} else {
		sampledIndex = Math.floor(Math.random() * treatment.numConfigs);
	}
	var configName = BASE_CONFIGURATION_PATH + treatment.experimentName;
	if (treatment.multiRound == 1) {
		configName += "/round_" + (treatment.roundNum).toString();
	}
	configName += "/game_json_" + sampledIndex.toString() + ".json";
	// configName += "/game_json_1.json";
	// my favorite testing config (remember to change back) please merge successfully

	console.log("use " + configName);
	configPath = basePath + configName;
	game.set("configFile", "public" + configName)

	// Define the bot treatment for the game
	game.set("botTreatment", treatment.botTreatment);

	// Number of turns each round
	game.set("numTurns", parseInt(treatment.numTurns));

	// Time for each turn (including speaking and listening)
	game.set("turnTime", parseInt(treatment.turnTime));

	// Time for the listener to choose
	game.set("listenTime", parseInt(treatment.listenTime));
	// console.log(game.get("numTurns"));

	game.set("annotation", treatment.annotation);

	// // An aray of possible turn times (list of integers)
	// var turnTimeArray = treatment.turnTimeList.split("_");
	// for (var i = 0; i < turnTimeArray.length; i++) {
	// 	turnTimeArray[i] = parseInt(turnTimeArray[i]);
	// }
	// game.set("turnTimeList", turnTimeArray);

	// An aray of numbers for constants in calculating bonus
	var bonusArray = treatment.bonusList.split("_");
	for (var i = 0; i < bonusArray.length; i++) {
		bonusArray[i] = parseInt(bonusArray[i]);
	}
	game.set("bonusList", bonusArray);

	// Load the game config json and the idx to tangram json
	let rawdata = fs.readFileSync(configPath);
	let gameFile = JSON.parse(rawdata);
	var gameConfig = gameFile["blocks"];

	var bonusData = gameFile["params"];
	game.set("pSimilarity", bonusData["p_similarity"])
	game.set("numRandom", bonusData["num_random"])

	// var idx2tPath = basePath + BASE_CONFIGURATION_PATH + "idx_to_tangram.json";
	// let idx2tRawdata = fs.readFileSync(idx2tPath);
	// let idx2t = JSON.parse(idx2tRawdata);

	// Iterate over each block
	var trialNum = 1;
	var allChatModes = new Set();
	for (let i = 0; i < gameConfig.length; i++) {
		// Create data for each tangram
		imageIndices = gameConfig[i]["img"];

		var imageDicts = Array(imageIndices.length);
		for (let j = 0; j < imageIndices.length; j++) {
			// imageIndex = imageIndices[j];
			// imageFile = idx2t[imageIndex.toString()];
			imageFile = imageIndices[j];
			imagePath = basePath + tangramsRelativeDir + imageFile;
			imageDicts[j] = {};
			imageDicts[j]["path"] = imageFile;
			imageDicts[j]["data"] = fs.readFileSync(imagePath, "utf8");
		}

		// Get the roles for the block
		roleArray = Array(2);
		currRoles = gameConfig[i]["roles"];
		for (let j = 0; j < 2; j++) {
			currRole = (currRoles[j] == 0) ? "speaker" : "listener";
			roleArray[j] = currRole;
		}

		const targetIndices = gameConfig[i]["tgt"];
		for (let j = 0; j < targetIndices.length; j++) {
			const round = game.addRound();
			round.set("chat", []);
			round.set("numTrials", gameConfig.length * targetIndices.length);
			round.set("trialNum", trialNum);
			trialNum++;

			round.set("numPartners", 1);
			round.set("tangrams", [
				getOrderedImages(imageDicts, gameConfig[i]["order"][0]),
				getOrderedImages(imageDicts, gameConfig[i]["order"][1]),
			]);
			round.set("numTangrams", imageIndices.length);

			// Changed to having N targets
			var targetPaths = [];
			for (let k = 0; k < targetIndices[j].length; k++) {
				var targetIndex = targetIndices[j][k];
				// targetPaths.push(idx2t[targetIndex.toString()]);
				targetPaths.push(targetIndex);
			}
			round.set("target", targetPaths);

			round.set("block", 0);
			round.set("controled", false);

			// round.set("roles", roleArray);
			// round.set("chatMode", "single-utterance-unidirectional");
			// allChatModes.add("single-utterance-unidirectional");
			round.set("chatMode", "multi-utterance-unidirectional");
			allChatModes.add("multi-utterance-unidirectional");
			round.set("clicks", []);
			round.set("clicked", false);
			round.set("turnCount", 0); // we want game.get("numTurns") turns at most
			// round.set("prevCorrect", false); // for displaying correct message

			// var turnTimeIndex = Math.floor(Math.random() * turnTimeArray.length)
			// round.set("turnTime", game.get("turnTimeList")[turnTimeIndex]);
			// console.log(round.get("turnTime"));
			round.set("timeList", []); // The time that has passed for turn i is l[i] (count start from 0)

			round.set("turnList", []); // A list that records information about the turn, including secUntilSend, secUntilSubmit (which is time for this turn), secBetweenSendAndSubmit, and clicks
			round.set("rightCount", 0); // The number of correct targets in the end
			round.set("wrongCount", 0); // If the same thing is selected wrongly twice, it gets counted twice
			round.set("forgiveSpeaker", 0); // number of times the speaker has been forgiven (so we need to add this to speaker message length)
			round.set("blamingWho", ""); // used for displaying social interactions
			// round.set("autoSubmitted", false); // so as not to have too many stage already submitted for listener auto submit
			// round.set("sawMessage", false); // if the listener has saw the speaker's message
			round.set("submitted", false);

			if (botGame) {
				console.log("in bot game");
				round.set("botActed", false);
				round.set('botActionTime', -1);

				playerIdx = (botIdx + 1) % 2
				// if (roleArray[0] == "listener") {
				// 	round.set("listener", game.players[playerIdx]);
				// 	round.set("speaker", game.players[botIdx]);
				// } else {
				round.set("speaker", game.players[playerIdx]);
				round.set("listener", game.players[botIdx]);
				game.players[playerIdx].set("role", "speaker");
				game.players[botIdx].set("role", "listener");
				// }
			} else {
				// Make expert speaker if in a expert-newbie pair
				if (game.players[0].get("expert") && !game.players[1].get("expert")) {
					round.set("speaker", game.players[0]);
					round.set("listener", game.players[1]);
					game.players[0].set("role", "speaker");
					game.players[1].set("role", "listener");
				} else if (game.players[1].get("expert") && !game.players[0].get("expert")) {
					round.set("speaker", game.players[1]);
					round.set("listener", game.players[0]);
					game.players[1].set("role", "speaker");
					game.players[0].set("role", "listener");
				} else if (!game.players[0].get("expert") && !game.players[1].get("expert")) {
					// both are newbies. Newbie with more points is speaker
					const sp_idx = game.players[0].get("points") > game.players[1].get("points") ? 0 : 1;
					const ls_idx = (sp_idx + 1) % 2;
					round.set("listener", game.players[ls_idx]);
					round.set("speaker", game.players[sp_idx]);
					game.players[ls_idx].set("role", "listener");
					game.players[sp_idx].set("role", "speaker");
				}
				else { // random if both experts
					if (roleArray[0] == "listener") {
						round.set("listener", game.players[0]);
						round.set("speaker", game.players[1]);
						game.players[0].set("role", "listener");
						game.players[1].set("role", "speaker");
					} else {
						round.set("speaker", game.players[0]);
						round.set("listener", game.players[1]);
						game.players[0].set("role", "speaker");
						game.players[1].set("role", "listener");
					}
				}
				console.log("expert " + game.players[0].get("expert") + " is " + game.players[0].get("role"));
				console.log("expert " + game.players[1].get("expert") + " is " + game.players[1].get("role"));
			}
			for (let i = 0; i < game.get("numTurns"); i++) {
				const stage = round.addStage({
					name: "turn-" + i,
					displayName: "turn—" + i,
					durationInSeconds: game.get("turnTime") + 10,
				});
			}

		}
	}

	game.set("allChatModes", allChatModes);
	game.set("consecutiveIdle", 0);
	game.set("updated", false);
	// game.set("disconnected", false);
	// console.log("finish initialize rounds");
});
