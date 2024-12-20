import React from "react";
import moment from "moment/moment";
import Author from "./Author";

function getDescriptionMessageDiv(player, round) {
  let message;
  if (round.get("chatMode") == "single-utterance-unidirectional") {
    if (player.get('role') == 'speaker') {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Please describe the object in the black box so your partner can
        correctly pick it out.<br />You can send only one message.<br />
        In addition, your partner cannot send messages at all.</div>);
    } else {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Click the object your partner is describing.<br />
        You cannot respond on chat.<br />Your partner can send only one message.</div>);
    }
  } else if (round.get("chatMode") == "multi-utterance-unidirectional") {// this is the part we are interested in
    if (player.get('role') == 'speaker') {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Please describe the object in the black box so your partner can
        correctly pick it out.<br />You can send one message each turn.<br />Your partner cannot send messages at all.</div>);
    } else {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Click the object your partner is describing.<br />
        You cannot respond on chat.<br />But your partner can send one message each turn.</div>);
    }
  } else if (round.get("chatMode") == "multi-utterance-backchannel") {
    if (player.get('role') == 'speaker') {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Please describe the object in the black box so your partner can
        correctly pick it out.<br />You can send many messages.<br />Your partner can send only a few emojis.</div>);
    } else {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Click the object your partner is describing.<br />
        You can respond on chat only by clicking the emojis.<br />However, your partner can send many messages.</div>);
    }
  } else if (round.get("chatMode") == "full-bidirectional") {
    if (player.get('role') == 'speaker') {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Please describe the object in the black box so your partner can
        correctly pick it out.<br />You and your partner can send many messages.</div>);
    } else {
      message = (<div style={{ "fontWeight": "normal" }}><strong>Instruction:</strong> Click the object your partner is describing.<br />
        Feel free to respond or ask questions if necessary.</div>);
    }
  }
  return message;
}
export default class EventLog extends React.Component {
  componentDidMount() {
    this.eventsEl.scrollTop = this.eventsEl.scrollHeight;
  }

  componentDidUpdate(prevProps) {
    if (prevProps.events.length < this.props.events.length) {
      this.eventsEl.scrollTop = this.eventsEl.scrollHeight;
    }
  }

  render() {
    const { game, events, player, round } = this.props;

    //if the one who made the event is the player himself then self will be true
    return (
      <div className="eventlog bp3-card">
        <div className="events" ref={el => (this.eventsEl = el)}>
          {events.map((event, i) => (
            <Event
              key={i}
              game={game}
              event={event}
              player={player}
              round={round}
              self={event.subject ? player._id === event.subject._id : null}
            />
          ))}
        </div>
      </div>
    );
  }
}

class Event extends React.Component {

  render() {
    const {
      subject,
      roundId,
      verb,
      object,
      target,
      state,
      at,
    } = this.props.event;
    const { self, round, game, player } = this.props;
    const partnerId = player.get('partner')
    const partner = _.filter(game.players, p => p._id === partnerId)[0];

    let content;
    switch (verb) {
      case "descriptionStarted":
        // console.log(getSelectionMessageDiv(round));
        content = (
          <>
            <div className="content">
              {getDescriptionMessageDiv(player, round)}
            </div>
            <br />
            <br />
          </>
        );
        break;

      case "feedbackStarted":
        content = (
          <div className="content">This is the feedback!
          </div>
        );
        break;

      default:
        console.error(`Unknown Event: ${verb}`);

        return null;
    }

    return (
      <div className="event">
        {/*
          Not sure we even need to show am/pm. I think we need seconds since the
          interactions are quick but to save space we can probably skip am/pm
          for the sake of space. We could maybe also just but the seconds since
          start of round or remaining second before end of round, might be more
          relevant. Might or might not be clear.
        */}
        {/* <div className="timestamp">{moment(at).format("hh:mm:ss a")}</div> */}
        <div className="timestamp">{moment(at).format("hh:mm:ss")}</div>
        {content}
      </div>
    );
  }
}
