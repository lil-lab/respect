import os
import random
import pickle

EMPTY_DATA_PATH = "data/dataset_splits2"

paths = os.listdir(EMPTY_DATA_PATH)
paths = [x[:-4] for x in paths]
if ".DS_S" in paths:
    paths.remove(".DS_S")

random.shuffle(paths)

dev = paths[:101]
train = paths[101:557]
test = paths[557:]

with open("dev_imgs.pkl", "wb") as f:
    pickle.dump(dev, f)

with open("train_imgs.pkl", "wb") as f:
    pickle.dump(train, f)

with open("test_imgs.pkl", "wb") as f:
    pickle.dump(test, f)
