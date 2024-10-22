import React from "react";

import { Centered, ConsentButton } from "meteor/empirica:core";
import BrowserDetection from "react-browser-detection";

export default class Consent extends React.Component {
  static renderConsent() {
    const searchParams = new URL(document.location).searchParams;
    const workerId = searchParams.get("workerId");
    const assignmentId = searchParams.get("assignmentId");

    return (
      <Centered>
        <div className="consent bp3-ui-text">
          <h1>Multi-reference Game Instructions</h1>
          <p>
            <b>
              For bug reports, early announcements about future HITs, and
              feedback join our DISCORD server:{" "}
              <a href="https://discord.gg/qyPsunaduU" target="_blank">
                <b>https://discord.gg/qyPsunaduU</b>
              </a>
              . You can also contact us at{" "}
              <a
                href="mailto:lillabcornell@gmail.com?subject=Multi-reference Game Question"
                target="_blank"
              >
                lillabcornell@gmail.com
              </a>
              .
            </b>
          </p>
          <h3 className="bp3-heading">TASK OVERVIEW</h3>
          <p>
            First time workers will be required to take a qualification quiz.
            The task itself consists of 3 stages (once you complete the short
            qualifier):{" "}
          </p>
          <ul>
            <li>
              You will enter a lobby to be matched with you partner. A sound
              will play after you get matched. If you do not get matched with a
              partner in 4 minutes, you will have the option of submitting the
              HIT.
            </li>
            <li>
              You will play 1 round of the Multi-reference game as the speaker.
            </li>
            <li>You will complete a survey, and submit the HIT.</li>
          </ul>
          <h3 className="bp3-heading">
            {" "}
            <a
              href="https://lil-lab.github.io/tangrams-multiref-dev/"
              target="_blank"
            >
              VIDEO DEMONSTRATION AND GAME RULES
            </a>{" "}
          </h3>
          <h3 className="bp3-heading">PAYMENT INFORMATION</h3>
          <p>
            A complete game usually takes between <b>4-6 minutes</b>. A game
            will take at most 10 turns and each turn is at most{" "}
            <b>70 seconds</b>.
          </p>
          <p>
            You will receive a <b>base hourly pay of $12.00/hr</b>, which is paid
            as bonus, for completing the game. There will be additional bonus
            if the listener selected more targets than non-targets, and extra 
            bonus when the game is successful (selecting all targets and no more). 
            With the success bonus, we expect an average hourly pay of $13.50/hr per HIT. 
            Players that we consider experts will have a 10% increase in their pay. 
            We track encrypted worker IDs to assign bonuses.
            Here is a <a href="/compensation_sample.png">compensation sample</a>. 
            Note that your bonus will vary.
          </p>

          <p>
            <b>Important: </b>
            Players who leave the game prior to completion or do not take an
            action within the time allotted to them{" "}
            <b>will receive no compensation</b>.
          </p>

          <p> 
            The payment associated with this AMT task is currently in{" "}
            <b>BETA</b>. <b>You may notice that we have increased the base pay as we are experimenting. This is temporary and subject to change.</b> As we gather data and continue to run the task, we will
            adjust the pay and compensate you accordingly.
          </p>
          <h3 className="bp3-heading">CONSENT</h3>
          <p>
            Before you choose to accept this HIT, please review our consent
            form:{" "}
            <a
              href="http://bit.ly/tangram-multiref-consent-form"
              target="_blank"
            >
              <b>http://bit.ly/tangram-multiref-consent-form</b>
            </a>
            {"."}
          </p>
          {workerId && assignmentId ? this.IAgreeButton() : ""}
        </div>
      </Centered>
    );
  }

  renderNoFirefox = () => {
    console.log("this is fire fox");
    return (
      <div className="consent">
        <h1
          className="bp3-heading"
          style={{ textAlign: "center", color: "red" }}
        >
          DO NOT USE FIREFOX!!
        </h1>
        <p style={{ textAlign: "center" }}>
          Please, don't use firefox! It breaks our game and ruins the experience
          for your potential teammates!
        </p>
      </div>
    );
  };

  static IAgreeButton = () => {
    return (
      <>
        <p>
          By clicking "I agree", you acknowledge that you are 18 years or older,
          have read this consent form, agree to its contents, and agree to take
          part in this research. If you do not wish to consent, close this page
          and return the task.
        </p>
        <ConsentButton text="I AGREE" />
      </>
    );
  };

  render() {
    const browserHandler = {
      default: (browser) =>
        browser === "firefox"
          ? this.renderNoFirefox()
          : Consent.renderConsent(),
    };

    return (
      <Centered>
        <BrowserDetection>{browserHandler}</BrowserDetection>
      </Centered>
    );
  }
}
