import React from "react";

import { Centered } from "meteor/empirica:core";
import { Button } from "@blueprintjs/core";

export default class OverviewTask extends React.Component {
  render() {
    const { hasPrev, hasNext, onNext, onPrev, treatment } = this.props;
    return (
      <Centered>
        <div className="instructions">
          <h1 className={"bp3-heading"}> Game Overview </h1>

          <h2> New Information</h2>

          <p>
            Following this screen, you will enter a lobby where you will be
            matched with a partner. If you cannot get paired for the game
            because the game is full or the waiting time is too long, it is
            important to <strong>submit the HIT</strong> so that you can receive
            the base pay for the task. If you choose to exit the lobby, you may
            need to refresh your page to see the submission button.
          </p>

          <p>
            After being matched with a partner, you will take part in{" "}
            <strong>6</strong> rounds of the reference game. Each round has{" "}
            {treatment.numTurns} turns, and each turn will take at most{" "}
            <strong>{treatment.turnTime}s</strong> total. In each turn, the
            speaker will have{" "}
            <strong>
              {parseInt(treatment.turnTime) - parseInt(treatment.listenTime)}s
            </strong>{" "}
            to describe the images. After the speaker's message, the listener
            will need to make a selection in the remaining timeframe and submit.
            The speaker will see green or red frames that indicate the
            listener's choice, and should continue to give descriptions of the
            images. The round is successful if the listener selects all the
            images correctly within the given turns.
          </p>

          <p>
            {/* You will receive a base bonus of $0.05 for each round you play.  */}
            For each round you play, you will receive an additional bonus based
            on a calculation from the number of correct and incorrect selections
            at the final turn, and the number of turns you took. While the
            number of successful rounds will be lower if you are paired with a
            fast but inaccurate partner, the quicker pace of rounds will ensure
            that you receive a similar hourly rate regardless of partner.
          </p>

          <h2>Tangrams Multiref Discord</h2>
          <p>
            In addition to giving feedback with the survey at the end of the
            game, we invite all participants to join our{" "}
            <a href="https://discord.gg/qyPsunaduU" target="_blank">
              discord server
            </a>
            . Joining this server is completely optional, but would help us keep
            track of bugs and receive real time feedback during the experiment.
            It also provides you with an opportunity to ask questions and learn
            about future HITs. If you choose to join, please make sure to read
            through the rules channel first.
          </p>

          <h2>Qualification Tutorial Review</h2>
          <p>
            In this task, you will play a series of games with a partner. In
            each round of the task, one player will be assigned a{" "}
            <strong>Speaker</strong> and the other a <strong>Listener</strong>{" "}
            role. Roles will alternate between rounds and a prompt on the screen
            will tell you your role. Remember, as long as the speaker types
            something in the chat log, their message will be auto-submitted when
            time is up. Likewise, if the listener makes a change to their
            selection, these changes will be auto-submitted when time is up. Of
            course, players are welcome to hit submit before time is up.
          </p>

          <p>
            As the <strong>speaker</strong>, your goal is to describe the target
            images to the listener by entering a message into a chatbox each
            turn. The amount of time you have in one turn will not be enough to
            describe all images and you will need to spread messages across
            turns effectively. Keep in mind that the fewer turns you take to
            successfully complete each round, the more bonuses you earn. Since
            the listener is shown images in a different order,{" "}
            <strong>you cannot use the position of the image</strong> in your
            description.
          </p>

          <p>
            As the <strong>listener</strong>, your goal is to select the images
            described by the speaker. If the listener selects all images
            correctly within the given turns, the round is a success.
          </p>

          <p>
            Both of you will see the same set of 20 abstract pictures. If you
            are the Speaker, 3-7 images will be highlighted with black frames.
            These are the <strong>target images.</strong> The listener does not
            see the black frames.
          </p>

          <div className="image-container">
            <center>
              <img
                width="300px"
                src="/experiment/examples_for_overview/listener_view.png"
              />
              <img
                width="300px"
                src="/experiment/examples_for_overview/speaker_view.png"
              />
            </center>
          </div>

          <p>
            After the listener submits, if the listener didn't select all the
            targets correctly or selected incorrect images, the speaker would
            see the listener's selections, with green showing it's indeed a
            target, and red showing it's incorrect. However, the listener won't
            be able to see this, and will learn what selections were correct and
            incorrect only through the speaker's messages.
          </p>

          <div className="image-container">
            <center>
              <img
                width="300px"
                src="/experiment/examples_for_overview/listener_wrong.png"
              />
              <img
                width="300px"
                src="/experiment/examples_for_overview/speaker_wrong.png"
              />
            </center>
          </div>

          <p>
            When the listener selects all images correctly, the listener and the
            speaker will both see the targets in green frames.
          </p>

          <div className="image-container">
            <center>
              <img
                width="300px"
                src="/experiment/examples_for_overview/listener_right.png"
              />
              <img
                width="300px"
                src="/experiment/examples_for_overview/speaker_right.png"
              />
            </center>
          </div>

          <p>
            Also, please limit your description to the current target pictures.
            Do <strong>not</strong> discuss previous rounds or chat about any
            other topics!
          </p>

          <button
            type="button"
            className="bp3-button bp3-intent-primary"
            onClick={onNext}
          >
            Next
            <span className="bp3-icon-standard bp3-icon-double-chevron-right bp3-align-right" />
          </button>
        </div>
      </Centered>
    );
  }
}
