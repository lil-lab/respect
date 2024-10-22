import moment from "moment";

import { GameLobbies } from "../game-lobbies.js";
import { Batches } from "../../batches/batches.js";
import { LobbyConfigs } from "../../lobby-configs/lobby-configs";
import { Players } from "../../players/players.js";
import { createGameFromLobby } from "../../games/create.js";
import Cron from "../../../startup/server/cron.js";

const sortLobbyByType = (weightedLobbyPool, type, currlobby) => {
  // Filter lobbies with type "expert"
  const sortedLobbies = weightedLobbyPool.filter(lobby => lobby.type === type && lobby.value._id !== currlobby._id);

  // Sort the expert lobbies by player time in descending order (longest first)
  sortedLobbies.sort((lobbyA, lobbyB) => {
    // console.log(Players.findOne({ _id: lobbyA.value.queuedPlayerIds[0] }))
    // console.log(new Date(Players.findOne({ _id: lobbyA.value.queuedPlayerIds[0] }).timeoutStartedAt))
    const startTimeAtA = new Date(Players.findOne({ _id: lobbyA.value.queuedPlayerIds[0] }).timeoutStartedAt);
    const startTimeAtB = new Date(Players.findOne({ _id: lobbyB.value.queuedPlayerIds[0] }).timeoutStartedAt);
    // Compare the startTimeAt values to determine the order
    return startTimeAtA - startTimeAtB;
  });

  return sortedLobbies;
}

const isExpert = (player) => {
  const hashedId = player.urlParams["workerId"];
  const matchingWorkers = Players.find({
    'urlParams.workerId': hashedId,
    'data.expert': true
  }).fetch();

  return !(!matchingWorkers || matchingWorkers.length === 0);
}

const typeOfLobby = (lobby) => {
  var type = "";
  const queuedPlayerList = lobby.queuedPlayerIds;
  for (let j = 0; j < queuedPlayerList.length; j += 1) {
    currPlayer = Players.findOne({ _id: queuedPlayerList[j] });
    if (currPlayer.bot === undefined) {
      if (isExpert(currPlayer)) {
        type = "expert";
      } else {
        type = "newbie";
      }
    }
  }
  return type;
}


const checkLobbyTimeout = (log, lobby, lobbyConfig) => {
  // Timeout hasn't started yet
  if (!lobby.timeoutStartedAt) {
    return;
  }

  const now = moment();
  const startTimeAt = moment(lobby.timeoutStartedAt);
  const endTimeAt = startTimeAt.add(lobbyConfig.timeoutInSeconds, "seconds");
  const ended = now.isSameOrAfter(endTimeAt);

  if (!ended) {
    return;
  }

  switch (lobbyConfig.timeoutStrategy) {
    case "fail":
      GameLobbies.update(lobby._id, {
        $set: { timedOutAt: new Date(), status: "failed" }
      });
      Players.update(
        { _id: { $in: lobby.queuedPlayerIds } },
        {
          $set: {
            exitStatus: "gameLobbyTimedOut",
            exitAt: new Date()
          }
        },
        { multi: true }
      );
      break;
    case "ignore":
      createGameFromLobby(lobby);
      break;

    // case "bots": {

    //   break;
    // }

    default:
      log.error(
        `unknown LobbyConfig.timeoutStrategy: ${lobbyConfig.timeoutStrategy}`
      );
  }
};

const checkIndividualTimeout = (log, lobby, lobbyConfig) => {
  const now = moment();
  Players.find({ _id: { $in: lobby.queuedPlayerIds } }).forEach(player => {
    const startTimeAt = moment(player.timeoutStartedAt);
    const timeSpentInSeconds = now.diff(startTimeAt, "seconds");
    const endTimeAt = startTimeAt.add(lobbyConfig.timeoutInSeconds, "seconds");

    const ended = now.isSameOrAfter(endTimeAt);

    if (!ended || player.timeoutWaitCount <= lobbyConfig.extendCount) {

      if (timeSpentInSeconds < 15 || timeSpentInSeconds % 5 != 0) { return; }

      // console.log("start looking to merge");

      // const batch = Batches.findOne(
      //   { status: "running", full: false },
      //   { sort: { runningAt: 1 } }
      // );
      const lobbies = GameLobbies.find({
        // batchId: batch._id,
        status: "running",
        timedOutAt: { $exists: false },
        gameId: { $exists: false }
      }).fetch();
      let lobbyPool = lobbies.filter(
        l => l.availableCount > l.queuedPlayerIds.length
      );

      const playerType = isExpert(player) ? "expert" : "newbie";
      const weightedLobbyPool = lobbyPool.map(lobby => {
        return {
          value: lobby,
          // weight: numHumansInQueue(lobby)
          type: typeOfLobby(lobby)
        };
      });

      sortedLobbyPool = sortLobbyByType(weightedLobbyPool, playerType, lobby);
      // console.log("sortedLobbyPool");
      // console.log(sortedLobbyPool);

      if (sortedLobbyPool.length > 0) {

        console.log("start merging");
        // quit from old
        GameLobbies.update(lobby._id, {
          $pull: {
            playerIds: player._id,
            queuedPlayerIds: player._id
          }
        });
        GameLobbies.update(lobby._id, {
          $addToSet: {
            quitPlayerIds: player._id
          }
        });

        // console.log("adding to new");
        // add to new
        GameLobbies.update(sortedLobbyPool[0].value._id, {
          $addToSet: {
            queuedPlayerIds: player._id,
            playerIds: player._id
          }
        });
        Players.update(player._id, {
          $set: {
            gameLobbyId: sortedLobbyPool[0].value._id
          }
        });
        // console.log(sortedLobbyPool[0].value);
        // console.log("merge completed");
      }
    }
    else {
      console.log("lobby time out");
      Players.update(player._id, {
        $set: {
          exitStatus: "playerLobbyTimedOut",
          exitAt: new Date(),
          exitReason: "Thanks for waiting, and sorry that there weren't enough other players for your game to being in a timely fashion!",
          ["data.lobbyEndAt"]: new Date(),
          ["data.bonus"]: 0.6
        }
      });
      GameLobbies.update(lobby._id, {
        $pull: {
          playerIds: player._id,
          queuedPlayerIds: player._id
        }
      });

      GameLobbies.update(lobby._id, {
        $addToSet: {
          quitPlayerIds: player._id
        }
      });
    }
  });
};

Cron.add({
  name: "Check lobby timeouts",
  interval: 1000,
  task: function (log) {
    const query = {
      status: "running",
      gameId: { $exists: false },
      timedOutAt: { $exists: false }
    };

    GameLobbies.find(query).forEach(lobby => {
      const lobbyConfig = LobbyConfigs.findOne(lobby.lobbyConfigId);

      switch (lobbyConfig.timeoutType) {
        case "lobby":
          checkLobbyTimeout(log, lobby, lobbyConfig);
          break;
        case "individual":
          checkIndividualTimeout(log, lobby, lobbyConfig);
          break;
        default:
          log.error(
            `unknown LobbyConfig.timeoutType: ${lobbyConfig.timeoutType}`
          );
      }
    });
  }
});
