import React from "react";

import { Centered } from "meteor/empirica:core";

import {
  Button,
  Classes,
  FormGroup,
  RadioGroup,
  TextArea,
  Intent,
  Radio,
} from "@blueprintjs/core";

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

export default class ErrorSurvey extends React.Component {
  static stepName = "ExitSurvey";
  state = {
    feedback: "",
  };

  handleChange = (event) => {
    const el = event.currentTarget;
    this.setState({ [el.name]: el.value });
  };

  handleSubmit = (event) => {
    event.preventDefault();

    const { player, game } = this.props;
    player.set("errorSurveyResponses", this.state);

    console.log(this.state);
    console.log(this.props.onSubmit);
    this.props.onSubmit(this.state);
  };

  surveyForm = (player, game) => {
    const { feedback } = this.state;
    let msg;
    switch (player.exitStatus) {
      case "gameFull":
        msg = "All games you are eligible for have filled up too fast...";
        break;
      case "gameLobbyTimedOut":
        msg = "There were NOT enough players for the game to start..";
        break;
      case "playerEndedLobbyWait":
        msg =
          "You decided to stop waiting, we are sorry it was too long a wait.";
        break;
      default:
        msg = "Unfortunately this game was cancelled...";
        break;
    }

    if (!(player.exitStatus == "custom") && (player.get("showHIT") == false)) {
      console.log("exitStatus: " + player.exitStatus);
      player.set("showHIT", true);
    }

    if (player.exitReason) msg = player.exitReason;
    return (
      <div>
        <h1>Your game has ended early</h1>
        {/* <p>{msg}</p> */}

        {player.get("blame") == 2 ? (
          <><div>
            <p>
              We terminated the game because we detected you idling for two turns. As described in the HIT preview page and the consent form (section Incentives for Participation), you will not get paid for this game.
            </p>
            <p>
              If you have any questions or concerns, you can reach out to us on Discord (<a href="https://discord.gg/qyPsunaduU" target="_blank">
                https://discord.gg/qyPsunaduU
              </a>) or via email at{" "}
              <a href="mailto: lillabcornell@gmail.com">
                lillabcornell@gmail.com
              </a>.

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
          <><p>{msg}</p>
            <form onSubmit={this.handleSubmit}>
              <h2>Error and bug reporting question</h2>

              <div className="pt-form-group">
                <div className="pt-form-content">
                  <FormGroup
                    className={"form-group"}
                    inline={false}
                    label={"Did you notice any problems or have any other comments about the study?"}
                    labelFor={"feedback"}
                  >
                    <TextArea
                      id="feedback"
                      name="feedback"
                      large={true}
                      intent={Intent.PRIMARY}
                      onChange={this.handleChange}
                      value={feedback}
                      fill={true} />
                  </FormGroup>
                </div>
              </div>
              <br />

              <button type="submit" className="pt-button pt-intent-primary">
                Submit
                <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
              </button>
            </form></>)}
      </div>
    );
  };

  componentWillMount() { }

    kickedOutMessage = (player) => {
	const msg = player.exitReason;
	return (
	    <>
	    <h1>{msg}</h1>

	    <p>
		We require players to complete our HITs one at a time. You cannot enter a game if you
		are already in the lobby for one or are already playing with someone else. If you have
		questions or concerns, please email us at{" "}
		<a href="mailto: lillabcornell@gmail.com">lillabcornell@gmail.com</a> or join{" "}
		<a href="https://discord.gg/qyPsunaduU" target="_blank">
		    our Discord server
		</a>.
	    </p>
	    </>
	);
    }


  render() {
    const { player, game } = this.props;
    let msg;
    switch (player.exitStatus) {
      case "gameFull":
        msg = "All games you are eligible for have filled up too fast...";
        break;
      case "gameLobbyTimedOut":
        msg = "There were NOT enough players for the game to start..";
        break;
      case "playerEndedLobbyWait":
        msg =
          "You decided to stop waiting, we are sorry it was too long a wait.";
        break;
      default:
        msg = "Unfortunately the Game was cancelled...";
        break;
    }
    if (player.exitStatus) msg = player.exitReason;

    if (player.exitStatus === "matchingPlayerKickedOut") {
	return (
	    <Centered>
		<div className="idle-error">{this.kickedOutMessage(player)}</div>
	    </Centered>
	);
	
    } else {
	return (
	    <Centered>
		<div className="exit-survey">{this.surveyForm(player, game)}</div>
	    </Centered>
	);
    }
  }
}
