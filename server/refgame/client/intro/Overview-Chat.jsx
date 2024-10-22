import React from "react";

import { Centered } from "meteor/empirica:core";
import { Button } from "@blueprintjs/core";
// import {useEffect} from 'react';

export default class OverviewChat extends React.Component {
  // for scroll up
  // constructor(props) {
  //   super(props)
  //   this.myRef = React.createRef()   // Create a ref object
  // }

  // componentDidMount() {
  //   window.scrollTo(0, 0);
  // }

  constructor(props) {
    super(props);
    this.myRef = React.createRef();
  }

  render() {
    const { hasPrev, hasNext, onNext, onPrev, player, treatment } = this.props;
    // useEffect(() => {
    //   // 👇️ scroll to top on page load
    //   window.scrollTo({top: 0, left: 0, behavior: 'smooth'});
    // }, []);
    return (
      <Centered>
        <div ref={this.myRef}></div> {/*Element to scroll to*/}
        <div className="instructions">
          <h1 className={"bp3-heading"}> Game Overview </h1>
          <h2>Chatbox and task's goal</h2>

          <p>
            All communication happens through a chat box on the left of the game
            interface.
          </p>

          <div className="image">
            <center>
              <img width="250px" src="/experiment/typing.PNG" />
            </center>
          </div>

          <p>
            You will be given one of four possible modes of text-box
            communication:
          </p>
          <p>
            <ol>
              <li>
                The speaker can send only <strong>one</strong> message (and the
                listener can't send anything).
              </li>
              <li>
                The speaker can send <strong>multiple</strong> messages (and the
                listener can't send anything).
              </li>
              <li>
                The speaker can send <strong>multiple</strong> messages, and the
                listener can send <strong>emojis</strong>:
                <div>
                  <div style={{ display: "inline-block" }}>
                    <img src="/experiment/emojis/x.PNG" />I am completely lost
                  </div>
                  <br />
                  <div style={{ display: "inline-block" }}>
                    <img src="/experiment/emojis/thinking.PNG" />I think I know
                    what you mean, but I need more information
                  </div>
                  <br />
                  <div style={{ display: "inline-block" }}>
                    <img src="/experiment/emojis/v.PNG" />I know exactly what
                    you mean
                  </div>
                  <br />
                  <div style={{ display: "inline-block" }}>
                    <img src="/experiment/emojis/laugh.PNG" />
                    Haha
                  </div>
                </div>
              </li>
              <li>
                Both speaker and listener can send <strong>multiple</strong>{" "}
                messages.
              </li>
            </ol>
          </p>

          <p>
            After the speaker sends a message (even when they can continue
            sending messages), the listener can select the target image by
            clicking on it. Once the listener clicks on an image, the round
            concludes.
          </p>
          <p>
            In each round, the speaker will have maximum <strong>45s</strong>{" "}
            seconds to describe the target, and the listener will have maximum{" "}
            <strong>60s</strong> to make the selection. In the last 15s of each
            round, the speaker will not be able to send any message while the
            listener is making their selection.
          </p>
          <p>
            Both partners will receive the feedback: the Speaker will see which
            picture the Listener clicked, and the Listener will see the true
            identity of the target. Depending on the communication mode, the
            listener may also send messages to the speaker.
          </p>

          <p>
            You will earn a $0.03 bonus for each correct match, so pay
            attention!
          </p>

          <p>
            If you cannot get paired for the game because the game is full or
            the waiting time is too long, it is important to{" "}
            <strong>return the submission</strong> so that we can compensate you
            with $1 for your time.
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
  executeScroll = () => this.myRef.current.scrollIntoView();
}
