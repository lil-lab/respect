# Guides for altering empirica source code

1. Open the directory `packages` under `refgame`
2. Clone `meteor-empirica-core` into `packages`
3. Copy `NewPlayer.jsx` into `ui/components/NewPlayer.jsx`
4. Copy `IdentifiedContainer.jsx` into `ui/containers/IdentifiedContainer.jsx`
5. Copy `methods.js` into `api/players/methods.js`
5. To launch meteor with changes, run

```
METEOR_PACKAGE_DIRS="packages" DEPLOY_HOSTNAME=galaxy.meteor.com meteor deploy tangrams-refgame.meteorapp.com --settings settings.json --owner lillab
```