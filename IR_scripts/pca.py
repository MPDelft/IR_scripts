import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from attr import asdict

from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, PADDED_Y_VALUE
from allrank.models.model import make_model


DATA_PATH = "/Users/marcus/PycharmProjects/allRank/MQ2007/Fold1"
CONFIG_PATH = "/Users/marcus/PycharmProjects/allRank/reproducibility/configs/wenjie.json"
MODEL_PATH = "/Users/marcus/PycharmProjects/allRank/models/rmse_sparse.pkl"
DATA_DIM = 46
TITLE = "RMSE Sparse"

train_ds, test_ds = load_libsvm_dataset(
    input_path=DATA_PATH,
    slate_length=240,
    validation_ds_role="test",
)
print(train_ds.shape)
print(test_ds.shape)

config = Config.from_json(CONFIG_PATH)

test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

n_features = train_ds.shape[-1]
model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
predictions = np.array([])
targets = np.array([])
x = np.array([np.zeros(DATA_DIM)])
with torch.no_grad():
    for xb, yb, indices in test_dl:
        mask = (yb == PADDED_Y_VALUE)
        p = model.score(xb, mask, indices).numpy().flatten()
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, yb.numpy().flatten()))

        s = xb.shape
        xx = xb.numpy()
        xx = xx.reshape((s[0] * s[1], s[2]))
        x = np.concatenate((x, xx), axis=0)

x = x[1:]
print("x shape:")
print(x.shape)
print("predictions")
print(predictions)
print("targets:")
print(targets)

unique, counts = np.unique(targets, return_counts=True)
print(dict(zip(unique, counts)))

min_pred = np.min(predictions)
max_pred = np.max(predictions)
range_pred = max_pred - min_pred
quarter = range_pred / 4

lower_b = min_pred + quarter
upper_b = max_pred - quarter

mid = (max_pred - min_pred) / 2

print(f"min: {min_pred}, max: {max_pred}")
print(f"lb: {lower_b}, ub: {upper_b}")

mapper = lambda x: -1 if x < lower_b else (2 if x > upper_b else (0 if x < mid else 1))

predictions = [mapper(x) for x in predictions]
correct = predictions == targets

print(len(correct))
print(correct)
print("Acc:", sum(correct)/len(correct))

pca = PCA(n_components=2)
pca_test = pca.fit_transform(x)

principal_Df = pd.DataFrame(data=pca_test, columns=['principal component 1', 'principal component 2'])

targets = [0, 1]
colors = ['r', 'g']

for target, color in zip(targets, colors):
    indicesToKeep = correct == target
    plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1'],
                principal_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

plt.title(TITLE)
plt.show()
