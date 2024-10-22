import React from "react";

import {Centered} from "meteor/empirica:core";

export default class MoreAboutBonus extends React.Component {
  render() {
    const { hasPrev, hasNext, onNext, onPrev, treatment } = this.props;
    const social = treatment.playerCount > 1;
    return (
      <Centered>
        <div className="instructions">
          <h1 className={"bp3-heading"}> Scores and Bonuses</h1>

          <p>
            In each task, we use "score" to evaluate the quality of the selections that you and your
            partner have made. Your total score will be calculated as the sum of the scores on each round.
          </p>

          <p>The score of your assignment is calculated as:</p>

          <div style={{ textAlign: "center" }}>
            <p>
              <strong>
                S = The number of correct responses * 0.03
              </strong>
            </p>
          </div>

          <p>
            Your performance bonus will be based on your score at the end of the experiment.
          </p>

          <p>
            <strong>
              {social ? "Remember, free riding is not permitted." : ""} If we
              detect that you are inactive during a task, you will not receive a
              bonus for that task.
            </strong>
          </p>

          <button
            type="button"
            className="bp3-button bp3-intent-nope bp3-icon-double-chevron-left"
            onClick={onPrev}
            disabled={!hasPrev}
          >
            Previous
          </button>
          <button
            type="button"
            className="bp3-button bp3-intent-primary"
            onClick={onNext}
            disabled={!hasNext}
          >
            Next
            <span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
          </button>
        </div>
      </Centered>
    );
  }
}
