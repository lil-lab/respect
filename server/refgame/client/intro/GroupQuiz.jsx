import React from "react";

import { Centered, AlertToaster } from "meteor/empirica:core";

import { Radio, RadioGroup } from "@blueprintjs/core";

import { Checkbox } from "@blueprintjs/core";

export default class GroupQuiz extends React.Component {
  state = {
    num_players: 0,
    teamColor: "",
    speakerGoal: "",
    listenerGoal: "",
    coloredTangrams: "",
    communication: "",
    chooseTarget: "",
    chooseDescription: "",
    roleSwap: ""
  };

  componentDidMount() {
    const { game } = this.props;
    document.querySelector("main").scrollTo(0, 0);
    this.state.num_players = game.treatment.playerCount + 2;
    this.state.teamColor = game.treatment.teamColor;
  }

  handleChange = (event) => {
    const el = event.currentTarget;
    this.setState({ [el.name]: el.value.trim().toLowerCase() });
  };

  handleRadioChange = (event) => {
    const el = event.currentTarget;
    console.log("el", el);
    console.log("ev", event);
    this.setState({ [el.name]: el.value });
  };

  handleEnabledChange = (event) => {
    const el = event.currentTarget;
    this.setState({ [el.name]: !this.state[el.name] });
  };

  handleSubmit = (event) => {
    event.preventDefault();

    //it should be this.state.nParticipants !== "3" but we don't have "treatment" in QUIZ
    if (
      this.state.speakerGoal !== "describe" ||
      this.state.listenerGoal !== "choose" ||
      this.state.communication !== "depend" ||
      this.state.roleSwap != "different" ||
      this.state.chooseTarget !== "A" ||
      this.state.chooseDescription !== "correct"

    ) {
      AlertToaster.show({
        message:
          "Sorry, you have one or more mistakes. Please ensure that you answer the questions correctly, or go back to the instructions",
      });
    } else {
      this.props.onNext();
    }
  };

  render() {
    const { hasPrev, onPrev, game, treatment } = this.props;
    return (
      <Centered>
        <div className="quiz">
          <h1 className={"bp3-heading"}> Quiz </h1>
          <form onSubmit={this.handleSubmit}>
            <div className="bp3-form-group"></div>

            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={<strong>1. As a speaker, what is your goal?</strong>}
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.speakerGoal}
                  name="speakerGoal"
                  required
                >
                  <Radio
                    label="Describe the target picture so the listener can easily pick it out from the other ones."
                    value="describe"
                  />
                  <Radio label="Get to know the listener." value="discuss" />
                  <Radio
                    label="Describe the location of the target picture on my screen (e.g. the one in the upper left corner)."
                    value="guess"
                  />
                  <Radio
                    label="Describe all the pictures on the screen."
                    value="all"
                  />
                </RadioGroup>
              </div>
            </div>

            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={<strong>2. As a listener, what is your goal?</strong>}
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.listenerGoal}
                  name="listenerGoal"
                  required
                >
                  <Radio
                    label="Describe the target picture I got."
                    value="describe"
                  />
                  <Radio
                    label="Chat with the speaker about your day."
                    value="discuss"
                  />
                  <Radio
                    label="Randomly click on a picture I like the most."
                    value="random"
                  />
                  <Radio
                    label="Choose the picture I think is closest to the speaker's description."
                    value="choose"
                  />
                </RadioGroup>
              </div>
            </div>
            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={
                    <strong>
                      3. How do the speaker and the listener communicate?
                    </strong>
                  }
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.communication}
                  name="communication"
                  required
                >
                  <Radio label="Both can send messages." value="both" />
                  <Radio
                    label="Only the speaker can send one message or many."
                    value="speaker"
                  />
                  <Radio
                    label="It depends on the chat mode. Each round may have its own chat mode."
                    value="depend"
                  />
                </RadioGroup>
              </div>
            </div>
            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={
                    <strong>
                      4. How do you know what your role is?
                    </strong>
                  }
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.roleSwap}
                  name="roleSwap"
                  required
                >
                  <Radio label="If you get speaker in the first round, you will be the speaker for the rest of the experiment." value="alwaysSpeaker" />
                  <Radio
                    label="Each round might be different. The prompt on the screen will tell you what your role is in this round."
                    value="different"
                  />
                  <Radio
                    label="You decide at the beginning of the experiment what your role is."
                    value="pick"
                  />
                </RadioGroup>
              </div>
            </div>
            <p>
              <strong>
                5. Consider the following description: 'A skater standing on one
                leg'.
              </strong>
            </p>
            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={
                    <strong>
                      Which of these tangrams do you think it best describes?
                    </strong>
                  }
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.chooseTarget}
                  name="chooseTarget"
                  inline={true}
                  required
                >
                  <Radio label="A" value="A" style={{ margin: "2%" }} />
                  <img src="/experiment/quiz_tangramA.png" height="200" />
                  <Radio label="B" value="B" style={{ margin: "2%" }} />
                  <img src="/experiment/quiz_tangramB.png" height="200" />
                  <Radio label="C" value="C" style={{ margin: "2%" }} />
                  <img src="/experiment/quiz_tangramC.png" height="200" />
                </RadioGroup>
              </div>
            </div>
            <p>
              <br />
              <br />
              <strong>
                6. Consider the following image of a target in a context:
              </strong>
            </p>
            <img src="/experiment/quiz_context.png" height="350" />
            <div className="bp3-form-group">
              <div className="bp3-form-content">
                <RadioGroup
                  label={
                    <strong>
                      Which of these descriptions would you prefer?
                    </strong>
                  }
                  onChange={this.handleRadioChange}
                  selectedValue={this.state.chooseDescription}
                  name="chooseDescription"
                  required
                >
                  <Radio label="the upper-left corner" value="false" />
                  <Radio
                    label="a square on top of some triangles"
                    value="false"
                  />
                  <Radio
                    label="a person walking to the right"
                    value="correct"
                  />
                </RadioGroup>
              </div>
            </div>
            {/* <div className="bp3-form-group">
              <label className="bp3-label" htmlFor="number-of-participants">
                If you do NOT choose a tangram before the time is up
                then your score will be:
              </label>
              <div className="bp3-form-content">
                <input
                  id="nParticipants"
                  className="bp3-input"
                  type="number"
                  min="-10"
                  max="10"
                  step="1"
                  dir="auto"
                  name="largeError"
                  value={this.state.largeError}
                  onChange={this.handleChange}
                  required
                />
              </div>
            </div> */}

            {/*<div className="bp3-form-group">*/}
            {/*  <label className="bp3-label" htmlFor="neighbor-of-room-101">*/}
            {/*    Which community have you been placed into?*/}
            {/*  </label>*/}
            {/*  <div className="bp3-form-content ">*/}
            {/*    <div className="bp3-control bp3-checkbox">*/}
            {/*      <Checkbox*/}
            {/*        name={"mc_2_101"}*/}
            {/*        label="Room 101"*/}
            {/*        onChange={this.handleEnabledChange}*/}
            {/*      />*/}
            {/*    </div>*/}
            {/*    <div className="bp3-control bp3-checkbox bp3-inline">*/}
            {/*      <Checkbox*/}
            {/*        name={"mc_2_102"}*/}
            {/*        label="Room 102"*/}
            {/*        onChange={this.handleEnabledChange}*/}
            {/*      />*/}
            {/*    </div>*/}
            {/*    <div className="bp3-control bp3-checkbox bp3-inline">*/}
            {/*      <Checkbox*/}
            {/*        name={"mc_2_103"}*/}
            {/*        label="Room 103"*/}
            {/*        onChange={this.handleEnabledChange}*/}
            {/*      />*/}
            {/*    </div>*/}
            {/*    <div className="bp3-control bp3-checkbox">*/}
            {/*      <Checkbox*/}
            {/*        name={"mc_2_104"}*/}
            {/*        label="Room 104"*/}
            {/*        onChange={this.handleEnabledChange}*/}
            {/*      />*/}
            {/*    </div>*/}
            {/*    <div className="bp3-control bp3-checkbox">*/}
            {/*      <Checkbox*/}
            {/*        name={"mc_2_105"}*/}
            {/*        label="Room 105"*/}
            {/*        onChange={this.handleEnabledChange}*/}
            {/*      />*/}
            {/*    </div>*/}
            {/*  </div>*/}
            {/*</div>*/}

            <button
              type="button"
              className="bp3-button bp3-intent-nope bp3-icon-double-chevron-left"
              onClick={onPrev}
              disabled={!hasPrev}
            >
              Back to instructions
            </button>
            <button type="submit" className="bp3-button bp3-intent-primary">
              Submit
              <span className="bp3-icon-standard bp3-icon-key-enter bp3-align-right" />
            </button>
          </form>
        </div>
      </Centered>
    );
  }
}
