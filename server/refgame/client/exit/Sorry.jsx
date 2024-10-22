import React from "react";

import { Centered } from "meteor/empirica:core";
import { Button } from "@blueprintjs/core";

const submitHIT = () => {
  const searchParams = new URL(document.location).searchParams;

  // create the form element and point it to the correct endpoint
  const form = document.createElement("form");
  form.action = new URL(
    "mturk/externalSubmit",
    searchParams.get("turkSubmitTo")
  ).href;
  form.method = "post";

  // attach the assignmentId
  const inputAssignmentId = document.createElement("input");
  inputAssignmentId.name = "assignmentId";
  inputAssignmentId.value = searchParams.get("assignmentId");
  inputAssignmentId.hidden = true;
  form.appendChild(inputAssignmentId);

  const inputCoordinates = document.createElement("input");
  inputCoordinates.name = "coordinates";
  inputCoordinates.value = "hello";
  inputCoordinates.hidden = true;
  form.appendChild(inputCoordinates);

  // attach the form to the HTML document and trigger submission
  document.body.appendChild(form);
  form.submit();
};

export default class Sorry extends React.Component {
  static stepName = "Sorry";

  render() {
    const { player, game, hasNext, onSubmit } = this.props;
    const blame = (player.get("blame") == 2);
    const showHIT = true; //player.get("showHIT");
    let bonus;
    if (player.get("bonus") === undefined) {
      bonus = 0.0;
    } else {
      bonus = player.get("bonus");
    }

    if (player.exitStatus) var exitMsg = player.exitReason;
    let payMsg;
    if (exitMsg && exitMsg.includes("stopped participating")) {
      payMsg =
        player.get("role") == "speaker"
          ? "as the speaker, you did not type anything within the allotted time twice"
          : "as the listener, you didn't submit your selection within the allotted time twice";
    } else if (exitMsg && exitMsg.includes("disconnected")) {
      payMsg = "you disconnected from the game";
    }

    // let msg;
    // switch (player.exitStatus) {
    //   case "gameFull":
    //     msg = "All games you are eligible for have filled up too fast...";
    //     break;
    //   case "gameLobbyTimedOut":
    //     msg = "There were NOT enough players for the game to start..";
    //     break;
    //   case "playerEndedLobbyWait":
    //     msg =
    //       "You decided to stop waiting, we are sorry it was too long a wait.";
    //     break;
    //   default:
    //     msg = "Unfortunately the Game was cancelled...";
    //     break;
    // }

    // if (player.exitReason) msg = player.exitReason;
    return (
      <Centered>
        <div className="score">
          <h1>Sorry!</h1>
          <br />
          {(!blame && showHIT) ? (
            <>
              <div>
                <p>
                  <strong>{exitMsg}</strong>
                </p>
                <p>
                  <strong>
                    Your final compensation is ${bonus.toFixed(2)}.
                  </strong>
                </p>
                <p>
                  You can reach out to us on Discord (<a href="https://discord.gg/qyPsunaduU" target="_blank">
                    https://discord.gg/qyPsunaduU
                  </a>) or via email at{" "}
                  <a href="mailto: lillabcornell@gmail.com">
                    lillabcornell@gmail.com
                  </a>{" "}
                  if you have any questions or concerns.
                </p>
              </div>

              <button
                type="button"
                className="bp3-button bp3-intent-primary"
                onClick={submitHIT}
              >
                Submit HIT
                <span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
              </button>
            </>
          ) : (
            <><div>
              <p>
                As described in the HIT preview page and in the incentives
                section of the consent form, you will <strong>not</strong>{" "}
                receive pay{payMsg !== undefined ? " because " : ""}
                <strong>{payMsg}</strong>.
              </p>
              <p>
                You can reach out to us on Discord (<a href="https://discord.gg/qyPsunaduU" target="_blank">
                  https://discord.gg/qyPsunaduU
                </a>) or via email at{" "}
                <a href="mailto: lillabcornell@gmail.com">
                  lillabcornell@gmail.com
                </a>{" "}
                if you have any questions or concerns.
              </p>
            </div>

              <button
                type="button"
                className="bp3-button bp3-intent-primary"
                onClick={submitHIT}
              >
                Submit HIT
                <span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
              </button>
            </>
          )}
        </div>
      </Centered>
    );
  }
}
