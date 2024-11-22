# Visualization and initial data processing

Execute `streamlit run app.py` will

1. Download data from MongoDB that was collected from server/refgame, and save as pickles.
2. Open a web app to visualize the game history

The pickles saved here contains multi-turn interactions,
and will be further broken down into turns in src/dataset/generate_dataset.py,
ready for training the policy for next round.
