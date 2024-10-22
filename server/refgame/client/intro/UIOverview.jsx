import React from "react";

import { Centered } from "meteor/empirica:core";
export default class UIOverview extends React.Component {
  render() {
    const { hasPrev, hasNext, onNext, onPrev, treatment } = this.props;
    const imagePath =
      treatment.playerCount > 1
        ? "experiment/interface.PNG"
        : "experiment/indUIExample.svg";

    console.log("imagePath", imagePath);

    return (
      <Centered>
        <div className="instructions">
          <h1 className={"bp3-heading"}> Game Interface</h1>
          <p>
            We are almost there! Please take a second to familiarize yourself
            with the game Interface shown here:
          </p>

          <div className="image">
            <img src={imagePath} style={{ border: "2px solid" }} />
          </div>

          <p>
            Now you know where everything goes and ready to take the quiz! Good
            luck.
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
