import React from "react";
import { Breadcrumb as Crumb, Classes } from "@blueprintjs/core";

export default class customBreadcrumb extends React.Component {
  render() {
    const { game, round, stage } = this.props;
    return (
      <nav className="round-nav">
        <ul className={Classes.BREADCRUMBS}>
          <li>
            {/* <Crumb
              text={"Round " + (round.get('trialNum')) +
                " / " + round.get('numTrials')}
              className={Classes.BREADCRUMB_CURRENT}
            />*/}
            <span style={{ margin: '0 1rem' }}></span>
            <Crumb
              text={"Turn " + Math.min(game.get('numTurns'), (round.get('turnCount') + 1)) +
                " / " + game.get('numTurns')}
              className={Classes.BREADCRUMB_CURRENT}
            />
          </li>
        </ul>
      </nav>
    );
  }
}
