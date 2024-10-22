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

// in the function it's promoting the player, and when this is called, we call it for the "partner"
const updateAndPromote = function (player) {
  const addPoints = player.get("addPoints");
  if (addPoints > 0) {
    player.set("points", player.get("points") + addPoints);
    console.log("points updated to: " + player.get("points"));
  }

  if (player.get("points") >= 3) {
    player.set("expert", true);
  }

}


export default class ExitSurvey extends React.Component {
  static stepName = "ExitSurvey";
  state = {
    satisfied: "",
    comprehension: "",
    grammatical: "",
    clear: "",
    ambiguous: "",
    english: "",
    languages: "",
    whereLearn: "",
    // fair: "",
    feedback: "",
    // chatUseful: "",
  };

  handleChange = (event) => {
    const el = event.currentTarget;
    this.setState({ [el.name]: el.value });
  };

  demographicHandleSubmitListener = (event) => {
    event.preventDefault();

    // Check if player responded to each required category
    const requiredCategories = [
      "satisfied",
      "grammatical",
      "clear",
      "ambiguous",
      "english",
      "languages",
      "whereLearn",
    ];
    let allowSubmit = true;
    for (let i = 0; i < requiredCategories.length; i += 1) {
      if (this.state[requiredCategories[i]] === "") {
        allowSubmit = false;
      }
    }

    if (allowSubmit) {
      const { player, game } = this.props;
      player.set("surveyResponses", this.state);
      player.set("completedDemographics", true);

      const partner = _.find(
        game.players,
        (p) => p._id === player.get("partner")
      );
      if (this.state["satisfied"] >= 4 && this.state["grammatical"] >= 4) {
        updateAndPromote(partner);
      }

      console.log(this.state);
      console.log(this.props.onSubmit);
      this.props.onSubmit(this.state);
    } else {
      console.log("Please answer the unanswered questions");
    }
  };

  demographicHandleSubmitSpeaker = (event) => {
    event.preventDefault();

    // Check if player responded to each required category
    const requiredCategories = [
      "satisfied",
      "comprehension",
      "english",
      "languages",
      "whereLearn",
    ];
    let allowSubmit = true;
    for (let i = 0; i < requiredCategories.length; i += 1) {
      if (this.state[requiredCategories[i]] === "") {
        allowSubmit = false;
      }
    }

    if (allowSubmit) {
      const { player, game } = this.props;
      player.set("surveyResponses", this.state);
      player.set("completedDemographics", true);

      const partner = _.find(
        game.players,
        (p) => p._id === player.get("partner")
      );
      if (this.state["satisfied"] >= 4 && this.state["comprehension"] >= 4) {
        updateAndPromote(partner);
      }

      console.log(this.state);
      console.log(this.props.onSubmit);
      this.props.onSubmit(this.state);
    } else {
      console.log("Please answer the unanswered questions");
    }
  };

  nonDemographicHandleSubmitListener = (event) => {
    event.preventDefault();

    // Check if player responded to each required category
    const requiredCategories = [
      "satisfied",
      "grammatical",
      "clear",
      "ambiguous",
    ];
    let allowSubmit = true;
    for (let i = 0; i < requiredCategories.length; i += 1) {
      if (this.state[requiredCategories[i]] === "") {
        console.log(requiredCategories[i]);
        allowSubmit = false;
      }
    }

    if (allowSubmit) {
      const { player, game } = this.props;
      player.set("surveyResponses", this.state);
      player.set("completedDemographics", true);

      const partner = _.find(
        game.players,
        (p) => p._id === player.get("partner")
      );
      if (this.state["satisfied"] >= 4 && this.state["grammatical"] >= 4) {
        updateAndPromote(partner);
      }

      console.log(this.state);
      console.log(this.props.onSubmit);
      this.props.onSubmit(this.state);
    } else {
      console.log("Please answer the unanswered questions");
    }
  };

  nonDemographicHandleSubmitSpeaker = (event) => {
    event.preventDefault();

    // Check if player responded to each required category
    const requiredCategories = ["satisfied", "comprehension"];
    let allowSubmit = true;
    for (let i = 0; i < requiredCategories.length; i += 1) {
      if (this.state[requiredCategories[i]] === "") {
        console.log(requiredCategories[i]);
        allowSubmit = false;
      }
    }

    if (allowSubmit) {
      const { player, game } = this.props;
      player.set("surveyResponses", this.state);
      player.set("completedDemographics", true);

      const partner = _.find(
        game.players,
        (p) => p._id === player.get("partner")
      );
      if (this.state["satisfied"] >= 4 && this.state["comprehension"] >= 4) {
        updateAndPromote(partner);
      }

      console.log(this.state);
      console.log(this.props.onSubmit);
      this.props.onSubmit(this.state);
    } else {
      console.log("Please answer the unanswered questions");
    }
  };

  renderGamePerformanceQuestionsListener = () => {
    const { satisfied, comprehension, grammatical, clear, ambiguous } =
      this.state;
    return (
      <div>
        <h2>Game performance questions</h2>
        <h3>
          You must answer questions in this section to be able to proceed.
        </h3>

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                1. How satisfied are you with your <b>partner's performance</b>{" "}
                in the game?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="satisfied"
              onChange={this.handleChange}
              selectedValue={satisfied}
            >
              <Radio label="Very satisfied" value="6" className={"pt-inline"} />
              <Radio label="Satisfied" value="5" className={"pt-inline"} />
              <Radio label="Somewhat satisfied" value="4" className={"pt-inline"} />
              <Radio label="Somewhat dissatisfied" value="3" className={"pt-inline"} />
              <Radio label="Dissatisfied" value="2" className={"pt-inline"} />
              <Radio label="Very dissatisfied" value="1" className={"pt-inline"} />
            </RadioGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                2. On a scale of 1-6 (where 6 is the best), how{" "}
                <b>grammatical</b> were your partner's descriptions?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="grammatical"
              onChange={this.handleChange}
              selectedValue={grammatical}
            >
              <Radio
                label="6 - there were no grammatical issues"
                value="6"
                className={"pt-inline"}
              />
              <Radio label="5" value="5" className={"pt-inline"} />
              <Radio label="4" value="4" className={"pt-inline"} />
              <Radio label="3" value="3" className={"pt-inline"} />
              <Radio label="2" value="2" className={"pt-inline"} />
              <Radio
                label="1 - almost all descriptions were ungrammatical"
                value="1"
                className={"pt-inline"}
              />
            </RadioGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                3. On a scale of 1-6 (where 6 is the easiest), how easy was it
                to <b>understand</b> your partner's descriptions?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="clear"
              onChange={this.handleChange}
              selectedValue={clear}
            >
              <Radio
                label="6 - all descriptions were easy to understand"
                value="6"
                className={"pt-inline"}
              />
              <Radio label="5" value="5" className={"pt-inline"} />
              <Radio label="4" value="4" className={"pt-inline"} />
              <Radio label="3" value="3" className={"pt-inline"} />
              <Radio label="2" value="2" className={"pt-inline"} />
              <Radio
                label="1 - no description was easy to understand"
                value="1"
                className={"pt-inline"}
              />
            </RadioGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                4. On a scale of 1-6 (where 6 is the easiest), how easily could
                you <b>distinguish</b> the target(s) from other images in the
                context, based on your partner's descriptions?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="ambiguous"
              onChange={this.handleChange}
              selectedValue={ambiguous}
            >
              <Radio
                label="6 - all targets were easy to distinguish"
                value="6"
                className={"pt-inline"}
              />
              <Radio label="5" value="5" className={"pt-inline"} />
              <Radio label="4" value="4" className={"pt-inline"} />
              <Radio label="3" value="3" className={"pt-inline"} />
              <Radio label="2" value="2" className={"pt-inline"} />
              <Radio
                label="1 - no targets were easy to distinguish"
                value="1"
                className={"pt-inline"}
              />
            </RadioGroup>
          </div>
        </div>
        <br />
      </div>
    );
  };

  renderGamePerformanceQuestionsSpeaker = () => {
    const { satisfied, comprehension, grammatical, clear, ambiguous } =
      this.state;
    return (
      <div>
        <h2>Game performance questions</h2>
        <h3>
          You must answer questions in this section to be able to proceed.
        </h3>

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                1. How satisfied are you with your <b>partner's performance</b>{" "}
                in the game?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="satisfied"
              // label="1. How satisfied are you with your partner's performance in the game?"
              onChange={this.handleChange}
              selectedValue={satisfied}
            >
              <Radio label="Very satisfied" value="6" className={"pt-inline"} />
              <Radio label="Satisfied" value="5" className={"pt-inline"} />
              <Radio label="Somewhat satisfied" value="4" className={"pt-inline"} />
              <Radio label="Somewhat dissatisfied" value="3" className={"pt-inline"} />
              <Radio label="Dissatisfied" value="2" className={"pt-inline"} />
              <Radio label="Very dissatisfied" value="1" className={"pt-inline"} />
            </RadioGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                2. On a scale of 1-6 (where 6 is the best), how well did your
                partner <b>understand your descriptions</b>?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="comprehension"
              // label="2. On a scale of 1-6 (where 6 is the best), how well did your partner understand your descriptions?"
              onChange={this.handleChange}
              selectedValue={comprehension}
            >
              <Radio
                label="6 - understood almost all descriptions"
                value="6"
                className={"pt-inline"}
              />
              <Radio label="5" value="5" className={"pt-inline"} />
              <Radio label="4" value="4" className={"pt-inline"} />
              <Radio label="3" value="3" className={"pt-inline"} />
              <Radio label="2" value="2" className={"pt-inline"} />
              <Radio
                label="1 - did not understand any of my descriptions"
                value="1"
                className={"pt-inline"}
              />
            </RadioGroup>
          </div>
        </div>
        <br />
      </div>
    );
  };

  renderDemographicQuestions = () => {
    const { english, languages, whereLearn } = this.state;

    return (
      <>
        <h2>Demographic questions</h2>
        <h3>
          You must answer questions in this section to be able to proceed. You
          will not be asked these demographic questions again in future HITs
          from this study.
        </h3>

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                1. Is English your native language?
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <RadioGroup
              name="english"
              // label="6. Is English your native language?"
              onChange={this.handleChange}
              selectedValue={english}
            >
              <Radio label="Yes" value="yes" className={"pt-inline"} />
              <Radio label="No" value="no" className={"pt-inline"} />
            </RadioGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                {`2. What languages do you know? How would you rate your proficiency in each language? (1=basic knowledge, 5=native speaker level) 
									Please format the response in the form of "Language(Proficiency)", e.g.: German(5), French(4).`}
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <FormGroup
              className={"form-group"}
              inline={false}
              // 					label={`7. What languages do you know? How would you rate your proficiency in each language? (1=basic knowledge, 5=native speaker level)
              //   Please format the response in the form of "Language(Proficiency)", e.g.: German(5), French(4).`}
              labelFor={"languages"}
            >
              <TextArea
                id="languages"
                large={true}
                intent={Intent.PRIMARY}
                onChange={this.handleChange}
                value={languages}
                fill={true}
                name="languages"
              />
            </FormGroup>
          </div>
        </div>
        <br />

        <div className="pt-form-group">
          <div className="pt-form-content">
            <div style={{ marginBottom: "15px" }}>
              <label>
                3. Where did you learn English? (If you are a native speaker,
                please indicate the country where you learned to speak.)
                <span style={{ color: "red", marginLeft: "5px" }}>*</span>
              </label>
            </div>
            <FormGroup
              className={"form-group"}
              inline={false}
              // label={
              // 	"8. Where did you learn English? (If you are a native speaker, please indicate the country where you learned to speak.)"
              // }
              labelFor={"whereLearn"}
            >
              <TextArea
                id="whereLearn"
                large={true}
                intent={Intent.PRIMARY}
                onChange={this.handleChange}
                value={whereLearn}
                fill={true}
                name="whereLearn"
              />
            </FormGroup>
          </div>
        </div>
        <br />
      </>
    );
  };

  renderFinalQuestions = () => {
    const { comments } = this.state;

    return (
      <>
        <h3>Any other comments?</h3>

        <div className="pt-form-group">
          <div className="pt-form-content">
            <FormGroup
              className={"form-group"}
              inline={false}
              labelFor={"comments"}
            >
              <TextArea
                id="comments"
                name="comments"
                large={true}
                intent={Intent.PRIMARY}
                onChange={this.handleChange}
                value={comments}
                fill={true}
              />
            </FormGroup>
          </div>
        </div>
      </>
    );
  };


  fullSurveyListener = (submitFunction) => {
    return (
      <div>
        <h1>Finally, please answer the following short survey.</h1>
        <form onSubmit={submitFunction}>
          {this.renderGamePerformanceQuestionsListener()}
          {this.renderDemographicQuestions()}
          {this.renderFinalQuestions()}
          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  fullSurveySpeaker = (submitFunction) => {
    return (
      <div>
        <h1>Finally, please answer the following short survey.</h1>
        <form onSubmit={submitFunction}>
          {this.renderGamePerformanceQuestionsSpeaker()}
          {this.renderDemographicQuestions()}
          {this.renderFinalQuestions()}
          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  partialSurveyListener = (submitFunction) => {
    return (
      <div>
        <h1>Finally, please answer the following short survey.</h1>
        <form onSubmit={submitFunction}>
          {this.renderGamePerformanceQuestionsListener()}
          {this.renderFinalQuestions()}
          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  partialSurveySpeaker = (submitFunction) => {
    return (
      <div>
        <h1>Finally, please answer the following short survey.</h1>
        <form onSubmit={submitFunction}>
          {this.renderGamePerformanceQuestionsSpeaker()}
          {this.renderFinalQuestions()}
          <button type="submit" className="pt-button pt-intent-primary">
            Submit
            <span className="pt-icon-standard pt-icon-key-enter pt-align-right" />
          </button>
        </form>
      </div>
    );
  };

  componentWillMount() { }

  render() {
    const { player, game } = this.props;
    const showDemographics = !player.get("completedDemographics");
    const isSpeaker = player.get("role") == "speaker";
    var submitFunction;
    var formContent;
    if (showDemographics) {
      submitFunction = isSpeaker
        ? this.demographicHandleSubmitSpeaker
        : this.demographicHandleSubmitListener;
      formContent = isSpeaker
        ? this.fullSurveySpeaker(submitFunction)
        : this.fullSurveyListener(submitFunction);
    } else {
      submitFunction = isSpeaker
        ? this.nonDemographicHandleSubmitSpeaker
        : this.nonDemographicHandleSubmitListener;
      formContent = isSpeaker
        ? this.partialSurveySpeaker(submitFunction)
        : this.partialSurveyListener(submitFunction);
    }
    // const submitFunction = (showDemographics) ? this.demographicHandleSubmit : this.nonDemographicHandleSubmit;
    // const formContent = (showDemographics) ? this.fullSurvey(submitFunction) : this.partialSurvey(submitFunction);

    return (
      <Centered>
        <div className="exit-survey">{formContent}</div>
      </Centered>
    );
  }
}
